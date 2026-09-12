"""LPI compatibility, data augmentation and entrypoint safety regression tests."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config
from diffusiondet.dataset_mapper import DiffusionDetDatasetMapper, build_transform_gen
from tools.lpi_common import ROOT, experiment, PROTOCOL, write_json
from tools.lpi_config_validation import validate_lpi_configs
from tools import lpi_train, lpi_evaluate


class LpiIntegrationTests(unittest.TestCase):
    def test_formal_configs_and_dataset_namespace(self):
        validate_lpi_configs(ROOT)
        from diffusiondet.datasets import DATASETS
        for split in ['train','val','test']:
            self.assertEqual(DATASETS['lpi_'+split],(f'LPI_COCO/annotations/instances_{split}.json','LPI_COCO'))
        for i in ['lpi-001','lpi-002','yolo-lpi-001','yolo-lpi-002','yolo-lpi-003']:
            _,cfg,run,w=experiment(i)
            self.assertIn('/lpi/',str(run))
            self.assertTrue(w.name in ['best.pt','model_best.pth'])

    def test_mapper_honors_no_flip_and_keeps_empty_annotations(self):
        cfg=get_cfg();add_diffusiondet_config(cfg);add_mobilenetv4_config(cfg)
        cfg.merge_from_file(str(ROOT/'configs/experiments/lpi/lpi-001.yaml'))
        self.assertEqual(len(build_transform_gen(cfg,True)),1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'noise.png'
            pixels=np.zeros((640,640,3),dtype=np.uint8);pixels[:,:128]=255
            Image.fromarray(pixels).save(path)
            out=DiffusionDetDatasetMapper(cfg,True)({'file_name':str(path),'width':640,'height':640,'annotations':[]})
            self.assertEqual(len(out['instances']),0)
            self.assertTrue(np.array_equal(out['image'].numpy().transpose(1,2,0),pixels))
        cfg.INPUT.RANDOM_FLIP='horizontal'
        self.assertEqual(len(build_transform_gen(cfg,True)),2)
        cfg.INPUT.RANDOM_FLIP='vertical'
        self.assertTrue(build_transform_gen(cfg,True)[0].vertical)
        cfg.INPUT.RANDOM_FLIP='invalid'
        with self.assertRaises(ValueError):build_transform_gen(cfg,True)

    def test_explicit_clean_git_option_blocks_dirty_training(self):
        with patch('sys.argv',['lpi_train.py','lpi-001','--require-clean-git']),patch.object(lpi_train,'git_state',return_value={'status':' M x','commit':'a'}),patch.object(lpi_train.subprocess,'run'),patch.object(lpi_train,'run_python') as run:
            with self.assertRaisesRegex(ValueError,'Git commit'):lpi_train.main()
            run.assert_not_called()

    def test_dirty_tree_allowed_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(lpi_train,'ROOT',Path(tmp)),patch('sys.argv',['lpi_train.py','lpi-001']),patch.object(lpi_train,'git_state',return_value={'status':' M x','commit':'a'}),patch.object(lpi_train.subprocess,'run'),patch.object(lpi_train,'run_python') as run:
                lpi_train.main()
                run.assert_called_once_with('Difdet','tools/lpi_train.py','lpi-001','--worker')

    def test_existing_training_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'log.txt').write_text('keep')
            with patch.object(lpi_train,'experiment',return_value=(d/'config.yaml',{},d,d/'model_best.pth')),patch.object(lpi_train,'check_environment',return_value={}),patch.object(lpi_train,'check_dataset',return_value={}):
                with self.assertRaisesRegex(ValueError,'拒绝覆盖'):lpi_train.worker('lpi-001')
                self.assertEqual((d/'log.txt').read_text(),'keep')

    def test_evaluation_missing_best_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'model_latest.pth').write_text('not best')
            with patch.object(lpi_evaluate,'ROOT',d),patch('sys.argv',['lpi_evaluate.py','lpi-001']),patch.object(lpi_evaluate,'experiment',return_value=(d/'config.yaml',{},d,d/'model_best.pth')),patch.object(lpi_evaluate,'run_python') as run:
                with self.assertRaisesRegex(ValueError,'最佳权重不存在'):lpi_evaluate.main()
                run.assert_not_called()

    def test_validation_calibration_precedes_test_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);w=d/'model_best.pth';w.write_text('checkpoint')
            with patch.object(lpi_evaluate,'ROOT',d),patch('sys.argv',['lpi_evaluate.py','lpi-001']),patch.object(lpi_evaluate,'experiment',return_value=(d/'config.yaml',{},d,w)),patch.object(lpi_evaluate,'run_python') as run:
                lpi_evaluate.main()
            calls=[c.args for c in run.call_args_list]
            self.assertEqual(calls[0][1],'tools/lpi_predict.py');self.assertIn('val',calls[0])
            self.assertEqual(calls[1][1:3],('tools/lpi_metrics.py','calibrate'))
            self.assertIn('test',calls[2]);self.assertEqual(calls[3][1:3],('tools/lpi_metrics.py','evaluate'))

if __name__=='__main__':unittest.main()
