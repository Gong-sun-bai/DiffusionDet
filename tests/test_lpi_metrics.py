import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.lpi_common import NAMES, PROTOCOL, digest, load_predictions, sha256, write_json
from tools.lpi_metrics import (matched_flags, build_statistics, counts, select_thresholds,
                               false_alarm, false_alarm_upper, snr_groups, coco_metrics,
                               compatible_thresholds, evaluate_external_noise)


def det(score=.8,category=1,box=None):
    return {'score':score,'category_id':category,'bbox':box or [0,0,10,10]}


def ann(category=1,box=None):
    return {'category_id':category,'bbox':box or [0,0,10,10]}


class LpiMetricsTests(unittest.TestCase):
    def test_duplicate_wrong_class_and_best_unmatched_iou(self):
        p=[det(.9,2),det(.8),det(.7)]
        _,flags=matched_flags(p,[ann()])
        self.assertEqual(flags.tolist(),[False,True,False])
        _,loc=matched_flags(p,[ann()],True)
        self.assertEqual(loc.tolist(),[True,False,False])
        _,flags=matched_flags([det(.9),det(.8,box=[5,0,10,10])],[ann(),ann(box=[5,0,10,10])])
        self.assertEqual(flags.tolist(),[True,True])

    def test_iou_half_and_threshold_inclusive(self):
        # Ground truth area 100, prediction area 200, intersection 100 -> exactly .5.
        p=[det(.5,box=[0,0,20,10])]
        s=build_statistics({1:{}},{1:[ann()]},{1:p})
        self.assertEqual(counts(s,[1],.5)['TP_obj'],1)
        self.assertEqual(counts(s,[1],math.nextafter(.5,math.inf))['TP_obj'],0)

    def test_noise_counts_windows_not_boxes(self):
        s=build_statistics({1:{},2:{},3:{}},{1:[],2:[],3:[ann()]},{1:[det()]*10,2:[],3:[det()]*8})
        result=false_alarm(s,[1,2,3],.5)
        self.assertEqual((result['FP_win'],result['TN_win'],result['N_noise']),(1,1,2))
        self.assertEqual(result['Pfa'],.5)

    def test_zero_false_alarms_not_certified(self):
        upper=false_alarm_upper(0,4950)
        self.assertAlmostEqual(upper,0.0006050153434647314)
        self.assertGreater(upper,1e-6)
        self.assertLess(false_alarm_upper(0,2995731),1e-6)
        self.assertEqual(false_alarm_upper(10,10),1)
        self.assertIsNone(false_alarm_upper(0,0))

    def test_cp_matches_scipy_one_sided(self):
        from scipy.stats import binomtest
        for k,n in [(0,10),(1,10),(9,10),(10,10)]:
            self.assertAlmostEqual(false_alarm_upper(k,n),binomtest(k,n,alternative='less').proportion_ci(method='exact').high)

    def test_f1_tie_higher_low_fa_strict_and_float64(self):
        s=build_statistics({1:{},2:{}},{1:[ann()],2:[]},{1:[det(.8)],2:[det(.5)]})
        points,_=select_thresholds(s)
        self.assertEqual(points['f1']['threshold'],.8)
        self.assertGreater(points['low_fa']['threshold'],.5)
        self.assertEqual(false_alarm(s,[1,2],points['low_fa']['threshold'])['FP_win'],0)
        self.assertEqual(counts(s,[1,2],points['low_fa']['threshold'])['TP_obj'],1)

    def test_low_fa_unavailable_and_empty_predictions(self):
        for score,expected in [(1.,None),(None,0.)]:
            s=build_statistics({1:{},2:{}},{1:[ann()],2:[]},{1:[],2:[] if score is None else [det(score)]})
            points,_=select_thresholds(s)
            self.assertEqual(points['low_fa']['threshold'],expected)
        self.assertEqual(counts(s,[1,2],.5)['Pd'],0)

    def test_bins_boundary_no_noise_and_no_interpolation(self):
        images={i:{'snr_db':s} for i,s in enumerate([-10,-8,0,8,10,None])}
        groups=snr_groups(images,[-10,0,10])
        bins=[g for g in groups if g[0].startswith('bin_')]
        self.assertEqual(bins[0][2],[0,1]);self.assertEqual(bins[1][2],[2,3,4])
        for bad in [[0,10],[-10,-10,10],[-10,float('nan'),10]]:
            with self.assertRaises(ValueError):snr_groups(images,bad)

    def test_ap_retains_low_scores_and_handles_empty(self):
        coco={'images':[{'id':1,'width':640,'height':640},{'id':2,'width':640,'height':640}],
              'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,10,10],'area':100,'iscrowd':0}],
              'categories':[{'id':i+1,'name':n} for i,n in enumerate(NAMES)]}
        out=coco_metrics(coco,{1:[det(.0001)],2:[]},[1,2])
        self.assertAlmostEqual(out['all']['AP'],1)
        out=coco_metrics(coco,{1:[],2:[]},[1,2])
        self.assertEqual(out['all']['AP'],0)
        self.assertIsNone(out['NLFM']['AP'])

    def test_threshold_identity_rejects_test_calibration_and_other_weights(self):
        identity={'weights_sha256':'w','config_sha256':'c','dataset_manifest_sha256':'d','implementation_sha256':'i','protocol':PROTOCOL}
        t={'calibration_split':'val','protocol':PROTOCOL,'identity':identity}
        compatible_thresholds(t,{'identity':identity})
        with self.assertRaises(ValueError):compatible_thresholds({**t,'calibration_split':'test'},{'identity':identity})
        with self.assertRaises(ValueError):compatible_thresholds(t,{'identity':{**identity,'weights_sha256':'wrong'}})

    def test_missing_duplicate_or_tampered_prediction_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'annotations').mkdir();bundle=root/'bundle';bundle.mkdir()
            coco={'images':[{'id':1,'file_name':'a.png','snr_db':0,'scene_id':'s'}, {'id':2,'file_name':'b.png','snr_db':None,'scene_id':'n'}],
                  'annotations':[{'id':1,'image_id':1,**ann()}], 'categories':[{'id':i+1,'name':n} for i,n in enumerate(NAMES)]}
            write_json(root/'annotations/instances_test.json',coco);write_json(root/'dataset_manifest.json',{})
            identity={'split':'test','annotations_sha256':sha256(root/'annotations/instances_test.json'),'dataset_manifest_sha256':sha256(root/'dataset_manifest.json')}
            def fixture(rows):
                (bundle/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                write_json(bundle/'manifest.json',{'status':'completed','identity':identity,'image_count':2,'predictions_sha256':sha256(bundle/'predictions.jsonl')})
            a={'image_id':1,'detections':[]};b={'image_id':2,'detections':[]}
            fixture([a,b]);self.assertEqual(len(load_predictions(bundle,root,'test')[0]),2)
            for rows in [[a],[a,a],[a,{'image_id':3,'detections':[]}]]:
                fixture(rows)
                with self.assertRaises(ValueError):load_predictions(bundle,root,'test')
            fixture([a,b]);(bundle/'predictions.jsonl').write_text('')
            with self.assertRaises(ValueError):load_predictions(bundle,root,'test')

    def test_external_noise_uses_frozen_scores_and_counts_each_window_once(self):
        identity={'config_sha256':'c','weights_sha256':'w','protocol':PROTOCOL,'implementation_sha256':'i'}
        thresholds={'identity':identity,'operating_points':{'f1':{'threshold':.5},'low_fa':{'threshold':.9}}}
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            rows=[{'window_id':'external-1','max_score':.5,'snr_db':None,'target_count':0},
                  {'window_id':'external-2','max_score':None,'snr_db':None,'target_count':0}]
            source=d/'noise_windows.jsonl'
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest={'kind':'independent_pure_noise','identity':identity,'source_manifest_sha256':'source',
                      'independence_description':'Independent seeds, disjoint from training/validation/test.',
                      'predictions_sha256':sha256(source),'window_count':2}
            write_json(d/'manifest.json',manifest)
            result=evaluate_external_noise(d,thresholds)
            self.assertEqual(result['operating_points']['f1']['FP_win'],1)
            self.assertEqual(result['operating_points']['low_fa']['FP_win'],0)
            self.assertFalse(result['operating_points']['low_fa']['Pfa_below_1e-6_supported'])
            rows[1]['window_id']='external-1'
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest['predictions_sha256']=sha256(source);write_json(d/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'重复'):evaluate_external_noise(d,thresholds)

    def test_match_prefix_statistics_equal_direct_rematching(self):
        rng=np.random.default_rng(41)
        pp=[det(float(rng.random()),int(rng.integers(1,3)),[float(rng.integers(0,12)),0,10,10]) for _ in range(50)]
        gt=[ann(),ann(2),ann(box=[10,0,10,10])]
        stats=build_statistics({1:{}},{1:gt},{1:pp})
        for t in [.01,.2,.5,.9,.99]:
            _,flags=matched_flags([p for p in pp if p['score']>=t],gt)
            self.assertEqual(counts(stats,[1],t)['TP_obj'],int(flags.sum()))

if __name__=='__main__':unittest.main()
