"""Framework-independent LPI matching, operating points and COCO/SNR reports."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.lpi_common import (NAMES, PROTOCOL, read_json, write_json, sha256, digest,
                              load_annotations, load_predictions)


def iou_xywh(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    a = np.asarray(box, dtype=np.float64)
    overlap = np.maximum(0, np.minimum(a[:2]+a[2:], boxes[:, :2]+boxes[:, 2:])-np.maximum(a[:2], boxes[:, :2]))
    intersection = overlap.prod(axis=1)
    union = a[2]*a[3] + boxes[:, 2]*boxes[:, 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def matched_flags(predictions, targets, ignore_class=False):
    """Stable score order, greedy maximum IoU among still-unmatched eligible GT."""
    ordered = sorted(predictions, key=lambda p: -p['score'])
    flags = np.zeros(len(ordered), dtype=bool)
    used = set()
    for j, p in enumerate(ordered):
        candidates = [k for k, t in enumerate(targets) if k not in used and (ignore_class or t['category_id'] == p['category_id'])]
        if not candidates:
            continue
        ious = iou_xywh(p['bbox'], [targets[k]['bbox'] for k in candidates])
        best = int(ious.argmax())
        if ious[best] >= .5:
            flags[j] = True
            used.add(candidates[best])
    return ordered, flags


def build_statistics(images, gt, predictions):
    """Match once. Every score-threshold prefix preserves these greedy assignments."""
    result = {}
    for i in sorted(images):
        ordered, cls_flags = matched_flags(predictions[i], gt[i])
        _, loc_flags = matched_flags(predictions[i], gt[i], ignore_class=True)
        result[i] = {'scores': np.array([p['score'] for p in ordered], dtype=np.float64),
                     'classes': np.array([p['category_id'] for p in ordered], dtype=np.int8),
                     'tp': cls_flags, 'loc_tp': loc_flags,
                     'gt_classes': np.array([t['category_id'] for t in gt[i]], dtype=np.int8)}
    return result


def counts(statistics, ids, threshold, category=None):
    tp = fp = n_gt = loc_tp = 0
    for i in ids:
        s = statistics[i]
        selected = s['scores'] >= threshold
        gt_selected = np.ones(len(s['gt_classes']), dtype=bool)
        if category is not None:
            selected &= s['classes'] == category
            gt_selected = s['gt_classes'] == category
        tp += int(s['tp'][selected].sum())
        fp += int(selected.sum()) - int(s['tp'][selected].sum())
        n_gt += int(gt_selected.sum())
        if category is None:
            loc_tp += int(s['loc_tp'][selected].sum())
    fn = n_gt - tp
    return {'TP_obj': tp, 'FP_obj': fp, 'FN_obj': fn, 'N_obj': n_gt,
            'Pd': tp/n_gt if n_gt else None,
            'Pd_localization': loc_tp/n_gt if n_gt and category is None else None,
            'F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def false_alarm_upper(k, n, confidence=.95):
    if n <= 0:
        return None
    if not 0 <= k <= n:
        raise ValueError('虚警计数超出范围')
    if k == 0:
        return -math.expm1(math.log1p(-confidence)/n)
    if k == n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(confidence, k+1, n-k))


def false_alarm(statistics, ids, threshold):
    noise = [i for i in ids if len(statistics[i]['gt_classes']) == 0]
    k = sum(bool(np.any(statistics[i]['scores'] >= threshold)) for i in noise)
    upper = false_alarm_upper(k, len(noise))
    return {'FP_win': k, 'TN_win': len(noise)-k, 'N_noise': len(noise),
            'Pfa': k/len(noise) if noise else None, 'Pfa_upper95': upper,
            'Pfa_below_1e-6_supported': upper is not None and upper < 1e-6,
            'confidence_method': 'one-sided 95% Clopper-Pearson',
            'evidence': '获得 Pfa < 10⁻⁶ 证据' if upper is not None and upper < 1e-6 else '无法证明 Pfa < 10⁻⁶'}


def select_thresholds(statistics):
    ids = list(statistics)
    grid = []
    for k in range(1, 100):
        t = k/100
        grid.append({'threshold': t, **counts(statistics, ids, t)})
    best = max(grid, key=lambda r: (r['F1'], r['threshold']))
    noise = [s for s in statistics.values() if len(s['gt_classes']) == 0]
    if not noise:
        low = {'threshold': None, 'reason': '验证集缺少纯噪声窗'}
    else:
        maxima = [float(s['scores'].max()) for s in noise if len(s['scores'])]
        t = math.nextafter(max(maxima), math.inf) if maxima else 0.0
        low = {'threshold': t if t <= 1 else None,
               'reason': '验证纯噪声最大分数严格后继值' if t <= 1 else '零虚警阈值超过 1，无可用工作点'}
    return {'f1': {'threshold': best['threshold'], 'validation_counts': best}, 'low_fa': low}, grid


def calibrate(data_root, bundle, output):
    coco, images, gt = load_annotations(data_root, 'val')
    predictions, manifest = load_predictions(bundle, data_root, 'val')
    statistics = build_statistics(images, gt, predictions)
    operating_points, grid = select_thresholds(statistics)
    for point in operating_points.values():
        if point['threshold'] is not None:
            point['validation_false_alarm'] = false_alarm(statistics, images, point['threshold'])
    result = {'schema_version': 1, 'calibration_split': 'val', 'identity': manifest['identity'],
              'validation_predictions_sha256': manifest['predictions_sha256'],
              'protocol': PROTOCOL, 'operating_points': operating_points, 'f1_grid': grid}
    if Path(output).exists():
        raise ValueError(f'阈值文件已存在，拒绝覆盖：{output}')
    write_json(output, result)
    return result


def snr_groups(images, edges=None):
    snrs = sorted({im['snr_db'] for im in images.values() if im['snr_db'] is not None})
    groups = [('overall', None, list(images))]
    groups += [(f'snr_{s:g}', s, [i for i, im in images.items() if im['snr_db'] == s]) for s in snrs]
    if edges is not None:
        if len(edges) < 2 or not all(math.isfinite(x) for x in edges) or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError('SNR 区间边界必须是至少两个严格递增有限数')
        if snrs and (edges[0] > min(snrs) or edges[-1] < max(snrs)):
            raise ValueError('SNR 区间必须覆盖全部已测 SNR')
        for j, (a, b) in enumerate(zip(edges, edges[1:])):
            last = j == len(edges)-2
            ids = [i for i, im in images.items() if im['snr_db'] is not None and a <= im['snr_db'] and (im['snr_db'] <= b if last else im['snr_db'] < b)]
            groups.append((f'bin_[{a:g},{b:g}{"]" if last else ")"}', None, ids))
    return groups


def coco_metrics(coco, predictions, ids):
    """Evaluate a group once, then derive class AP from the same COCO precision tensor."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    names = ['all', *NAMES]
    if not ids:
        return {n: {'AP': None, 'AP50': None, 'AP75': None} for n in names}
    id_set = set(ids)
    with contextlib.redirect_stdout(io.StringIO()):
        ground = COCO()
        ground.dataset = {'images': [im for im in coco['images'] if im['id'] in id_set],
                          'annotations': [a for a in coco['annotations'] if a['image_id'] in id_set],
                          'categories': coco['categories'], 'info': {}}
        ground.createIndex()
        rows = [dict(p, image_id=i) for i in ids for p in predictions[i]]
        if rows:
            detected = ground.loadRes(rows)
        else:
            detected = COCO()
            detected.dataset = {**ground.dataset, 'annotations': []}
            detected.createIndex()
        evaluator = COCOeval(ground, detected, 'bbox')
        evaluator.params.imgIds = sorted(ids)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval['precision'][:, :, :, 0, 2]
    output = {}
    for index, name in enumerate(names):
        values = precision if index == 0 else precision[:, :, index-1:index]
        metrics = {}
        for label, selected in [('AP', values), ('AP50', values[0:1]), ('AP75', values[5:6])]:
            valid = selected[selected >= 0]
            metrics[label] = float(valid.mean()) if valid.size else None
        output[name] = metrics
    return output


def compatible_thresholds(thresholds, manifest):
    if thresholds.get('calibration_split') != 'val' or thresholds.get('protocol') != PROTOCOL:
        raise ValueError('阈值必须来自验证集且使用当前协议')
    a, b = thresholds['identity'], manifest['identity']
    for key in ('config_sha256', 'weights_sha256', 'dataset_manifest_sha256', 'protocol', 'implementation_sha256'):
        if a[key] != b[key]:
            raise ValueError(f'阈值与测试预测不兼容：{key}')


def evaluate(data_root, bundle, thresholds_path, output, edges=None, external_noise=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    coco, images, gt = load_annotations(data_root, 'test')
    predictions, manifest = load_predictions(bundle, data_root, 'test')
    thresholds = read_json(thresholds_path)
    compatible_thresholds(thresholds, manifest)
    statistics = build_statistics(images, gt, predictions)
    groups = snr_groups(images, edges)
    rows = []
    background = {}
    for name, point in thresholds['operating_points'].items():
        t = point['threshold']
        background[name] = false_alarm(statistics, images, t) if t is not None else {'evidence': point['reason'], 'Pfa_below_1e-6_supported': False}
    for group, snr, ids in groups:
        print(f'COCOeval: {group}, {len(ids)} images', flush=True)
        ap = coco_metrics(coco, predictions, ids)
        for category, class_name in [(None, 'all'), *enumerate(NAMES, 1)]:
            for op, point in thresholds['operating_points'].items():
                t = point['threshold']
                c = counts(statistics, ids, t, category) if t is not None else {key: None for key in ['TP_obj', 'FP_obj', 'FN_obj', 'N_obj', 'Pd', 'Pd_localization', 'F1']}
                rows.append({'group': group, 'snr_db': snr, 'class': class_name, 'operating_point': op,
                             'threshold': t, 'N_images': len(ids), **ap[class_name], **c,
                             'joint_supported': c['Pd'] is not None and c['Pd'] >= .9 and background[op]['Pfa_below_1e-6_supported']})
    minimum = {}
    for op in thresholds['operating_points']:
        minimum[op] = {}
        for name in ['all', *NAMES]:
            subset = [r for r in rows if r['snr_db'] is not None and r['class'] == name and r['operating_point'] == op]
            pd90 = [r['snr_db'] for r in subset if r['Pd'] is not None and r['Pd'] >= .9]
            joint = [r['snr_db'] for r in subset if r['joint_supported']]
            minimum[op][name] = {'minimum_measured_snr_pd90': min(pd90) if pd90 else None,
                                 'minimum_measured_snr_joint_supported': min(joint) if joint else None}
    result = {'schema_version': 1, 'metric_scale': 'AP/Pd/Pfa in [0,1]', 'identity': manifest['identity'],
              'predictions_sha256': manifest['predictions_sha256'], 'thresholds_sha256': sha256(thresholds_path),
              'snr_bin_edges': edges, 'background': background, 'minimum_snr': minimum, 'rows': rows,
              'limitations': ['最低 SNR 仅指离散实测档位，不保证更高档均达标。',
                             '固定单训练种子；不证明跨种子稳定或真实接收环境泛化。',
                             '背景虚警以独立纯噪声观测窗为单位，不能分配 SNR。']}
    if external_noise:
        result['external_noise'] = evaluate_external_noise(external_noise, thresholds)
    write_json(output / 'metrics.json', result)
    write_json(output / 'thresholds.json', thresholds)
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'background.csv').open('w', newline='', encoding='utf-8-sig') as f:
        f.write('operating_point,threshold,FP_win,TN_win,N_noise,Pfa,Pfa_upper95\n')
        for op, b in background.items():
            f.write(','.join(str(v) for v in [op, thresholds['operating_points'][op]['threshold'], *[b.get(k, '') for k in ['FP_win','TN_win','N_noise','Pfa','Pfa_upper95']]])+'\n')
    render_report(result, output)
    return result


def evaluate_external_noise(path, thresholds):
    """Stream independent max-score windows; never pool with internal test or recalibrate."""
    import json
    manifest_path = Path(path) / 'manifest.json'
    manifest = read_json(manifest_path)
    a, b = thresholds['identity'], manifest['identity']
    for key in ('config_sha256', 'weights_sha256', 'protocol', 'implementation_sha256'):
        if a[key] != b[key]:
            raise ValueError(f'外部噪声预测身份不兼容：{key}')
    if manifest.get('kind') != 'independent_pure_noise' or not manifest.get('source_manifest_sha256') or not manifest.get('independence_description'):
        raise ValueError('外部噪声必须声明来源指纹和独立性依据')
    source = Path(path) / 'noise_windows.jsonl'
    if sha256(source) != manifest['predictions_sha256']:
        raise ValueError('外部噪声预测哈希不一致')
    n = 0
    seen = set()
    hits = {op: 0 for op in thresholds['operating_points']}
    with source.open() as f:
        for line in f:
            r = json.loads(line)
            wid = r['window_id']
            if not isinstance(wid, str) or wid in seen or not wid or r.get('snr_db') is not None or r.get('target_count') != 0:
                raise ValueError('外部噪声窗重复或元数据错误')
            seen.add(wid)
            score = r['max_score']
            if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('外部噪声分数非法')
            n += 1
            for op, p in thresholds['operating_points'].items():
                hits[op] += int(score is not None and p['threshold'] is not None and score >= p['threshold'])
    if n != manifest['window_count'] or n <= 0:
        raise ValueError('外部噪声计数不一致或为空')
    return {'source_manifest_sha256': sha256(manifest_path), 'independence_description': manifest['independence_description'],
            'note': '独立性由外部生成清单提供；未与原测试集合并，不改变原联合达标结论。',
            'operating_points': {op: ({'N_noise': n, 'FP_win': k, 'TN_win': n-k, 'Pfa': k/n,
                                     'Pfa_upper95': false_alarm_upper(k,n),
                                     'Pfa_below_1e-6_supported': false_alarm_upper(k,n) < 1e-6}
                                    if thresholds['operating_points'][op]['threshold'] is not None else None) for op,k in hits.items()}}


def render_report(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = ['# LPI 雷达检测评价', '', 'AP、Pd、Pfa 均以 0～1 表示；所有工作阈值只由验证集选择。', '',
             '| 工作点 | FP窗 / 噪声窗 | Pfa | 单侧95%上限 | 结论 |', '|---|---:|---:|---:|---|']
    for op,b in result['background'].items():
        lines.append(f"| {op} | {b.get('FP_win','—')} / {b.get('N_noise','—')} | {b.get('Pfa','—')} | {b.get('Pfa_upper95','—')} | {b['evidence']} |")
    lines += ['', '## 每档 SNR（五类总体）', '', '| 工作点 | SNR/dB | AP | AP50 | AP75 | Pd | 定位Pd | 联合达标证据 |', '|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(x):
        return '—' if x is None else f'{x:.6f}'
    for r in result['rows']:
        if r['snr_db'] is not None and r['class'] == 'all':
            lines.append('| '+ ' | '.join([r['operating_point'], str(r['snr_db']), *[fmt(r[k]) for k in ['AP','AP50','AP75','Pd','Pd_localization']], '是' if r['joint_supported'] else '否'])+' |')
    lines += ['', '## 达到 90% 检测率的最低已测 SNR', '']
    for op, classes in result['minimum_snr'].items():
        for name,m in classes.items():
            lines.append(f"- {op} / {name}：Pd≥90% 为 {m['minimum_measured_snr_pd90']} dB；联合证据为 {m['minimum_measured_snr_joint_supported']} dB（None 表示未获得）。")
    if 'external_noise' in result:
        lines += ['', '## 独立外部纯噪声评价', '', result['external_noise']['note']]
        for op, value in result['external_noise']['operating_points'].items():
            lines.append(f'- {op}：{value}')
    lines += ['', '每类与合并 SNR 区间的完整结果见 metrics.csv / metrics.json。', '', *result['limitations'], '',
              '置信限方法：https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for op in result['background']:
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        for name in ['all', *NAMES]:
            rr = [r for r in result['rows'] if r['snr_db'] is not None and r['class']==name and r['operating_point']==op]
            for ax,key in zip(axes, ['Pd','AP']):
                ax.plot([r['snr_db'] for r in rr], [np.nan if r[key] is None else r[key] for r in rr], marker='o', label=name)
                ax.set(xlabel='SNR (dB)',ylabel=key,ylim=(0,1.02));ax.grid(alpha=.3)
        axes[0].axhline(.9,color='k',linestyle='--',label='Pd=90%')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'LPI / {op} (validation-frozen threshold)')
        fig.tight_layout()
        for suffix in ['png','pdf']:
            fig.savefig(output/f'snr_{op}.{suffix}',dpi=180)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['calibrate','evaluate'])
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--thresholds',type=Path)
    p.add_argument('--snr-bin-edges',type=float,nargs='+')
    p.add_argument('--external-noise',type=Path)
    a=p.parse_args()
    if a.action=='calibrate':
        calibrate(a.data_root,a.predictions,a.output)
    else:
        if a.thresholds is None:p.error('evaluate 必须指定 --thresholds')
        evaluate(a.data_root,a.predictions,a.thresholds,a.output,a.snr_bin_edges,a.external_noise)

if __name__=='__main__':
    main()
