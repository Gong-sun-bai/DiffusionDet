"""One command: validation inference -> frozen thresholds -> test inference -> SNR report."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.lpi_common import (ROOT, DATA, CONFIGS, experiment, timestamp, run_python,
                              prediction_identity, load_predictions, write_json, read_json)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment_id',choices=CONFIGS)
    p.add_argument('--reuse',type=Path,help='已有完整评价目录；核验指纹后复用 val/test 预测，重新生成报告')
    p.add_argument('--snr-bin-edges',type=float,nargs='+')
    p.add_argument('--external-noise',type=Path)
    a=p.parse_args()
    config,cfg,run,weights=experiment(a.experiment_id)
    if not weights.is_file():raise ValueError(f'最佳权重不存在：{weights}')
    import fcntl
    lock=ROOT/'.cache/lpi_gpu.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=run/'evaluations'/timestamp();output.mkdir(parents=True,exist_ok=False)
        bundles={}
        env='yolo' if a.experiment_id.startswith('yolo-') else 'Difdet'
        for split in ['val','test']:
            bundle=(a.reuse.resolve() if a.reuse else output)/split
            if a.reuse and (a.reuse/'evaluation.json').is_file():
                bundle=Path(read_json(a.reuse/'evaluation.json')['bundles'][split])
            if a.reuse:
                # Coverage and source validation runs without importing either training framework.
                load_predictions(bundle,DATA,split,prediction_identity(config,weights,DATA,split))
            else:
                run_python(env,'tools/lpi_predict.py','--config',config,'--weights',weights,
                           '--data-root',DATA,'--split',split,'--output',bundle)
            bundles[split]=str(bundle)
            if split=='val':
                run_python('Difdet','tools/lpi_metrics.py','calibrate','--data-root',DATA,
                           '--predictions',bundle,'--output',output/'thresholds.json')
        flags=[]
        if a.snr_bin_edges:flags+=['--snr-bin-edges',*map(str,a.snr_bin_edges)]
        if a.external_noise:flags+=['--external-noise',str(a.external_noise.resolve())]
        run_python('Difdet','tools/lpi_metrics.py','evaluate','--data-root',DATA,'--predictions',bundles['test'],
                   '--thresholds',output/'thresholds.json','--output',output/'report',*flags)
        write_json(output/'evaluation.json',{'status':'completed','experiment_id':a.experiment_id,
                                          'bundles':bundles,'weights':str(weights)})
        print(f'评价完成：{output}/report/report.md')

if __name__=='__main__':main()
