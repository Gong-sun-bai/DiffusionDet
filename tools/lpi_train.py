"""Run one LPI experiment, with explicit environment, provenance and resume guards."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.lpi_common import (ROOT, DATA, CONFIGS, experiment, git_state, run_python, digest,
                              sha256, write_json, read_json, check_dataset, resolved_config)


def check_environment(yolo=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用')
    result = {'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0)}
    if yolo:
        import ultralytics
        result['ultralytics'] = ultralytics.__version__
        if torch.__version__ != '2.9.1+cu128' or ultralytics.__version__ != '8.4.91':
            raise RuntimeError(f'YOLO 环境版本不符合冻结口径：{result}')
    else:
        import torchvision, fvcore, pycocotools, timm, detectron2
        from detectron2 import _C
        result.update(torchvision=torchvision.__version__, timm=timm.__version__, detectron2=detectron2.__file__, extension=_C.__file__)
        if torch.__version__ != '2.10.0+cu128' or Path(detectron2.__file__).resolve().parent != ROOT/'detectron2':
            raise RuntimeError(f'Difdet 环境不符合冻结口径：{result}')
    free,total = torch.cuda.mem_get_info()
    result.update(free_memory_mb=free/1024**2, total_memory_mb=total/1024**2)
    if free < 20*1024**3:
        raise RuntimeError(f'可用显存不足 20 GiB，停止启动：{result}')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def train_yolo(config, resume=False, preflight=None):
    import torch
    from ultralytics import YOLO
    from tools.lpi_yolo import LpiDetectionTrainer
    cfg = resolved_config(Path(config))
    run = ROOT / cfg['project'] / cfg['name']
    config_hash = digest(cfg)
    record = run/'lpi_training.json'
    previous = read_json(record) if record.exists() else {}
    if resume:
        if previous.get('status') not in {'running','failed','interrupted'} or previous.get('config_sha256') != config_hash:
            raise ValueError('YOLO 恢复要求同配置且未完成的 LPI 训练记录')
        weight = run/'weights/last.pt'
        if not weight.is_file():
            raise ValueError('缺少 weights/last.pt')
        model = YOLO(str(weight))
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError(f'已有结果，拒绝覆盖：{run}')
        # Download official initialization into an ignored, stable cache directory.
        source = ROOT/'.cache/lpi_pretrained'/Path(cfg['model']).name
        source.parent.mkdir(parents=True,exist_ok=True)
        model = YOLO(str(source))
        previous = {'initial_weights': str(source), 'initial_weights_sha256': sha256(source),
                    'config_sha256': config_hash, 'git': git_state()}
    torch.cuda.reset_peak_memory_stats()
    def started(trainer):
        # Ultralytics otherwise silently halves batch after first-epoch OOM.
        trainer._oom_retries = 3
        actual = {'optimizer': type(trainer.optimizer).__name__, 'accumulate': trainer.accumulate,
                  'nominal_batch_size': trainer.args.nbs, 'batch': trainer.batch_size,
                  'train_input_shape': list(trainer.train_loader.dataset[0]['img'].shape),
                  'val_input_shape': list(trainer.test_loader.dataset[0]['img'].shape),
                  'train_transforms': str(trainer.train_loader.dataset.transforms),
                  'source_cache_policy': 'repository .cache/lpi_yolo_labels',
                  'optimizer_groups': [{k:v for k,v in group.items() if k != 'params'} for group in trainer.optimizer.param_groups]}
        write_json(record,{**previous,'status':'running','actual_training':actual})
        if preflight and not (run/'lpi_preflight.json').exists():
            from tools.lpi_source_snapshot import persist_preflight
            persist_preflight(preflight,run/'lpi_preflight.json')
        if not (run/'config.yaml').exists():
            (run/'config.yaml').write_text(Path(config).read_text(),encoding='utf-8')
    def epoch_end(trainer):
        if record.exists():
            r=read_json(record)
            r['last_epoch']=trainer.epoch
            r['last_accumulate']=trainer.accumulate
            r['peak_memory_mb']=torch.cuda.max_memory_allocated()/1024**2
            write_json(record,r)
    def fixed_batch(trainer):
        trainer._oom_retries = 3  # Native initialization resets this after on_train_start.
        if trainer.batch_size != cfg['batch']:
            raise RuntimeError('LPI 禁止自动修改 Batch')
    model.add_callback('on_train_batch_start',fixed_batch)
    model.add_callback('on_train_start',started)
    model.add_callback('on_train_epoch_end',epoch_end)
    try:
        if resume:
            model.train(trainer=LpiDetectionTrainer, resume=True, augmentations=[])
        else:
            args={k:v for k,v in cfg.items() if k not in {'model','mode','task'}}
            args['project']=str((ROOT/cfg['project']).resolve())
            args['data']=str((ROOT/cfg['data']).resolve())
            args['augmentations']=[]
            model.train(trainer=LpiDetectionTrainer, **args)
        r=read_json(record)
        r.update(status='completed',peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2,
                 best_weights_sha256=sha256(run/'weights/best.pt'))
        write_json(record,r)
    except BaseException as error:
        if record.exists():
            r=read_json(record);r.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));write_json(record,r)
        raise


def worker(experiment_id,resume=False):
    from tools.lpi_common import timestamp
    config,cfg,run,_=experiment(experiment_id)
    yolo=experiment_id.startswith('yolo-')
    environment=check_environment(yolo)
    source=check_dataset()
    identity={'config_sha256':digest(cfg),'data':source}
    provenance=run/'lpi_preflight.json'
    if run.exists() and any(run.iterdir()):
        if not resume:
            raise ValueError(f'已有非空结果目录，拒绝覆盖：{run}')
        if not provenance.exists() or any(read_json(provenance).get(k)!=v for k,v in identity.items()):
            # JSON tuples serialize as lists: compare canonical hashes below instead.
            old=read_json(provenance) if provenance.exists() else {}
            if digest({k:old.get(k) for k in identity})!=digest(identity):
                raise ValueError('恢复的配置或数据指纹发生变化')
        if not yolo:
            state=read_json(run/'experiment_manifest.json')
            if state.get('status')=='completed':raise ValueError('已完成实验不能恢复')
            checkpoint=run/'last_checkpoint'
            if not checkpoint.is_file() or not (run/checkpoint.read_text().strip()).is_file():
                raise ValueError('恢复 checkpoint 不存在')
    elif resume:
        raise ValueError('恢复目录不存在或为空')
    # Keep provenance outside the run until the native engine has passed its nonempty-directory check.
    preflight=ROOT/'.cache/lpi_preflight'/f'{experiment_id}-{timestamp()}.json'
    preflight.parent.mkdir(parents=True,exist_ok=True)
    from tools.lpi_source_snapshot import capture_source
    snapshot=capture_source(preflight.with_suffix(''))
    write_json(preflight,{**identity,'environment':environment,'git':git_state(),'source_snapshot':snapshot})
    try:
        if yolo:
            train_yolo(config,resume,preflight)
        else:
            command=[sys.executable,str(ROOT/'tools/lpi_diffusion_train.py'),'--preflight',str(preflight),'--num-gpus','1','--config-file',str(config)]
            if resume:command.append('--resume')
            subprocess.run(command,cwd=ROOT,check=True)
    finally:
        if run.exists():
            from tools.lpi_source_snapshot import persist_preflight
            if not provenance.exists():persist_preflight(preflight,provenance)
            else:persist_preflight(preflight,run/f'lpi_preflight-{timestamp()}.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment_id',choices=CONFIGS)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check-only',action='store_true',help='检查环境、配置和数据，不启动训练、不要求干净 Git')
    p.add_argument('--require-clean-git',action='store_true',help='可选：要求提交全部修改后才训练；默认允许先验证再提交')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        if a.check_only:
            check_environment(a.experiment_id.startswith('yolo-'));print(check_dataset())
        else:worker(a.experiment_id,a.resume)
        return
    subprocess.run([sys.executable,str(ROOT/'tools/validate_experiment_configs.py')],cwd=ROOT,check=True)
    if not a.check_only and git_state()['status']:
        if a.require_clean_git:
            raise ValueError('--require-clean-git 要求工作区修改已形成 Git commit。')
        print('工作区尚未提交：允许先训练验证；本次运行会保存源码快照和 Git 状态。',flush=True)
    # One user-selected experiment only; flock prevents concurrent LPI GPU jobs.
    import fcntl
    lock=ROOT/'.cache/lpi_gpu.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        flags=['--worker']+(['--resume'] if a.resume else [])+(['--check-only'] if a.check_only else [])
        run_python('yolo' if a.experiment_id.startswith('yolo-') else 'Difdet','tools/lpi_train.py',a.experiment_id,*flags)

if __name__=='__main__':
    main()
