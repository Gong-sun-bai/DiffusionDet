"""One explicitly selected LPI model: tiny train, empty/mixed backward, export and metrics smoke.

Never trains the five formal experiment IDs. Example:
  PYTHONNOUSERSITE=1 /home/troy/anaconda3/envs/Difdet/bin/python tools/smoke_lpi.py lpi-001 --test-id lpi-test-001
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from tools.lpi_common import (ROOT,DATA,CONFIGS,experiment,load_annotations,write_json,
                              sha256,digest,read_json,run_python,SEED)


def subset(output, train_size=32, eval_size=8):
    if output.exists():
        if not (output/'dataset_manifest.json').exists():raise ValueError('已有 smoke 数据未完成')
        return
    output.mkdir(parents=True,exist_ok=False)
    if train_size==32 and eval_size==8:
        quotas={'train':(24,8),'val':(6,2),'test':(6,2)}
    else:
        if train_size<32 or eval_size<8:raise ValueError('smoke size 太小')
        quotas={split:(n-max(1,round(n*.1)),max(1,round(n*.1))) for split,n in [('train',train_size),('val',eval_size),('test',eval_size)]}
    for split,(n_signal,n_noise) in quotas.items():
        coco,images,gt=load_annotations(DATA,split)
        signal=[i for i in images if gt[i]]
        # Cover low/high SNR, all modulation classes, and multi-object scenes.
        selected=[]
        for c in range(1,6):
            for snr in [-10,10]:
                candidates=[i for i in signal if images[i]['snr_db']==snr and any(a['category_id']==c for a in gt[i])]
                for i in candidates:
                    if i not in selected:selected.append(i);break
        selected=selected[:n_signal]
        spread=sorted(signal,key=lambda i:hashlib.sha256(f'lpi-smoke:{i}'.encode()).digest())
        for i in spread:
            if len(selected)>=n_signal:break
            if i not in selected:selected.append(i)
        selected += [i for i in images if not gt[i]][:n_noise]
        sub={**coco,'images':[images[i] for i in sorted(selected)],'annotations':[a for a in coco['annotations'] if a['image_id'] in selected]}
        (output/'annotations').mkdir(exist_ok=True)
        write_json(output/'annotations'/f'instances_{split}.json',sub)
        for i in selected:
            im=images[i];source=DATA/im['file_name'];dest=output/im['file_name'];dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest)
            label=Path('labels')/split/(source.stem+'.txt');(output/label).parent.mkdir(parents=True,exist_ok=True);os.link(DATA/label,output/label)
    (output/'dataset.yaml').write_text(yaml.safe_dump({'path':str(output),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(['LFM','NLFM','BPSK','BFSK','Frank']))}))
    write_json(output/'dataset_manifest.json',{'status':'completed','preview':True,'kind':'training_smoke_subset',
               'source_manifest_sha256':sha256(DATA/'dataset_manifest.json'),
               'annotations':{s:sha256(output/'annotations'/f'instances_{s}.json') for s in quotas}})


def diffusion_train(config,data_root):
    import torch
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.data.datasets import register_coco_instances
    for split in ['train','val']:
        register_coco_instances(f'lpi_smoke_{split}',{},str(data_root/'annotations'/f'instances_{split}.json'),str(data_root))
    args=default_argument_parser().parse_args(['--num-gpus','1','--config-file',str(config)])
    args.run_until_iter=None
    torch.cuda.reset_peak_memory_stats()
    from tools.lpi_diffusion_train import main as provenance_main
    from unittest.mock import patch
    preflight=Path(config).parent/'preflight.json'
    from tools.lpi_source_snapshot import capture_source
    snapshot=capture_source(Path(config).parent/'source_snapshot')
    write_json(preflight,{'smoke':True,'source_manifest_sha256':sha256(data_root/'dataset_manifest.json'),'source_snapshot':snapshot})
    with patch.object(sys,'argv',['lpi_diffusion_train.py','--preflight',str(preflight),'--config-file',str(config),'--num-gpus','1']):
        provenance_main()
    peak=torch.cuda.max_memory_allocated()/1024**2
    gc.collect();torch.cuda.empty_cache()
    return peak


def backward_diagnostics(config,data_root,weights):
    import torch
    from tools.lpi_predict import diffusion_config
    from diffusiondet.dataset_mapper import DiffusionDetDatasetMapper
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.structures import Boxes,Instances
    cfg=diffusion_config(config,weights)
    model=build_model(cfg);DetectionCheckpointer(model).load(str(weights));model.train()
    mapper=DiffusionDetDatasetMapper(cfg,is_train=True)
    _,images,gt=load_annotations(data_root,'train')
    batch_size=cfg.SOLVER.IMS_PER_BATCH
    noise_pool=[i for i in images if not gt[i]]
    noise=[noise_pool[j%len(noise_pool)] for j in range(batch_size)]
    positive=[i for i in images if gt[i]][:batch_size//2]
    report={}
    for name,ids in [('all_noise',noise),('mixed',noise[:batch_size-len(positive)]+positive)]:
        batch=[]
        for i in ids:
            im=images[i]
            annotations=[{**a,'bbox_mode':1,'category_id':a['category_id']-1} for a in gt[i]]
            batch.append(mapper({**im,'file_name':str(data_root/im['file_name']),'annotations':annotations}))
        model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        loss=model(batch);total=sum(loss.values())
        if not torch.isfinite(total):raise RuntimeError(f'{name} nonfinite loss')
        total.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError(f'{name} nonfinite gradient')
        if not any(g.abs().sum()>0 for g in grads):raise RuntimeError('背景分类梯度为零')
        report[name]={'losses':{k:float(v.detach()) for k,v in loss.items()},'finite_gradients':True,
                      'nonzero_gradient':True,'peak_memory_mb':torch.cuda.max_memory_allocated()/1024**2}
    del model,batch,loss,total,grads
    gc.collect();torch.cuda.empty_cache()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment_id',choices=CONFIGS)
    p.add_argument('--test-id',required=True)
    p.add_argument('--train-size',type=int,default=32)
    p.add_argument('--eval-size',type=int,default=8)
    p.add_argument('--iterations',type=int,default=2,help='DiffusionDet 迭代数')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--formal-optimizer',action='store_true',help='YOLO 小规模复核采用正式 200 epoch 的 MuSGD 配方')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();yolo=a.experiment_id.startswith('yolo-')
    pattern=r'yolo-lpi-test-\d{3}' if yolo else r'lpi-test-\d{3}'
    if not re.fullmatch(pattern,a.test_id):raise ValueError(f'test ID 必须匹配 {pattern}')
    if not a.worker:
        import fcntl
        lock=ROOT/'.cache/lpi_gpu.lock';lock.parent.mkdir(exist_ok=True)
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run_python('yolo' if yolo else 'Difdet','tools/smoke_lpi.py',a.experiment_id,'--test-id',a.test_id,'--worker','--train-size',a.train_size,'--eval-size',a.eval_size,'--iterations',a.iterations,'--workers',a.workers,*(['--formal-optimizer'] if a.formal_optimizer else []))
        return
    from tools.lpi_train import check_environment,train_yolo
    environment=check_environment(yolo)
    data_name='data' if (a.train_size,a.eval_size)==(32,8) else f'data-t{a.train_size}-e{a.eval_size}'
    data_root=ROOT/'.cache/lpi_smoke'/data_name;subset(data_root,a.train_size,a.eval_size)
    _,cfg,_,_=experiment(a.experiment_id)
    work=ROOT/'.cache/lpi_smoke'/a.test_id;work.mkdir(exist_ok=False)
    run=(ROOT/'runs/yolo/lpi' if yolo else ROOT/'runs/lpi')/(a.test_id+'__img640-short-smoke')
    if run.exists():raise ValueError(f'smoke 目录已存在：{run}')
    if yolo:
        cfg.update(name=run.name,project=str(run.parent),data=str(data_root/'dataset.yaml'),epochs=1,workers=a.workers,save_json=False,plots=False)
        if a.formal_optimizer:cfg.update(optimizer='MuSGD',momentum=.9,lr0=.01,warmup_bias_lr=0.)
    else:
        cfg['EXPERIMENT'].update(ID=a.test_id,NAME='img640-short-smoke',DESCRIPTION=f'{a.experiment_id} 结构的 640/Batch{cfg["SOLVER"]["IMS_PER_BATCH"]}、{a.iterations} iter 独立冒烟，不进入正式排名。')
        cfg['DATASETS']={'TRAIN':'("lpi_smoke_train",)','TEST':'("lpi_smoke_val",)'}
        cfg['DATALOADER']['NUM_WORKERS']=a.workers
        cfg['SOLVER'].update(MAX_ITER=a.iterations,STEPS='()',WARMUP_ITERS=0,CHECKPOINT_PERIOD=a.iterations)
        cfg['TEST']['EVAL_PERIOD']=a.iterations
        cfg['TRAINING_PLOTS']={'ENABLED':False}
    config=work/'config.yaml';config.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    started=time.monotonic()
    if yolo:
        from tools.lpi_source_snapshot import capture_source
        preflight=work/'preflight.json'
        write_json(preflight,{'smoke':True,'source_snapshot':capture_source(work/'source_snapshot')})
        train_yolo(config,preflight=preflight)
        weights=run/'weights/best.pt'
        details=read_json(run/'lpi_training.json')
    else:
        peak=diffusion_train(config,data_root)
        weights=run/'model_best.pth'
        details={'training_peak_memory_mb':peak,'backward':backward_diagnostics(config,data_root,weights)}
    output=run/'evaluations/smoke';output.mkdir(parents=True,exist_ok=False)
    env='yolo' if yolo else 'Difdet'
    # Release the worker model before new processes take the GPU.
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    for split in ['val','test']:
        run_python(env,'tools/lpi_predict.py','--config',config,'--weights',weights,'--data-root',data_root,'--split',split,'--output',output/split)
        if split=='val':run_python('Difdet','tools/lpi_metrics.py','calibrate','--data-root',data_root,'--predictions',output/split,'--output',output/'thresholds.json')
    run_python('Difdet','tools/lpi_metrics.py','evaluate','--data-root',data_root,'--predictions',output/'test','--thresholds',output/'thresholds.json','--output',output/'report')
    write_json(run/'smoke_summary.json',{'status':'completed','source_experiment':a.experiment_id,'test_id':a.test_id,
               'environment':environment,'details':details,'elapsed_seconds':time.monotonic()-started,
               'warning':'仅短训练运行验证，指标不是正式模型效果。'})
    print(f'Smoke completed: {run}',flush=True)

if __name__=='__main__':main()
