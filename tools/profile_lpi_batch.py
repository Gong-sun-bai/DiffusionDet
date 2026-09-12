"""Measure one model/batch in a fresh process: real LPI data, FP32 forward/backward/step.

This is a capacity/throughput probe, not a formal training experiment. Each model profile
uses a new test ID and stores per-batch results without overwriting earlier probes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.lpi_common import ROOT,DATA,CONFIGS,experiment,load_annotations,write_json,SEED


def sample_ids(images,gt,batch,offset=0,stress=False):
    positives=sorted(i for i in images if len(gt[i])==5) if stress else sorted(i for i in images if gt[i])
    noise=sorted(i for i in images if not gt[i])
    # Evenly spaced real scenes/SNRs across train; all-five-target stress is a separate phase.
    n_noise=0 if stress else max(1,round(batch*.1))
    return [positives[((j*7919)+offset*1543)%len(positives)] for j in range(batch-n_noise)]+[noise[((j*37)+offset*41)%len(noise)] for j in range(n_noise)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment_id',choices=CONFIGS)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--warmup',type=int,default=4)
    a=p.parse_args()
    if a.batch<=0 or a.steps<2 or a.warmup<1:raise ValueError('Invalid probe sizes')
    import fcntl
    lock=ROOT/'.cache/lpi_gpu.lock';lock.parent.mkdir(exist_ok=True)
    lock_handle=lock.open('a')
    fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    import numpy as np
    torch.manual_seed(SEED);np.random.seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    config,c,_,_=experiment(a.experiment_id)
    yolo='model' in c
    coco,images,gt=load_annotations(DATA,'train')
    gpu_samples=[];done=threading.Event()
    def monitor():
        while not done.is_set():
            try:
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().split(',')
                gpu_samples.append((time.monotonic(),float(value[0]),float(value[1])))
            except (OSError,subprocess.SubprocessError,ValueError):pass
            done.wait(.2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    result={'profile_protocol_version':2,'experiment_id':a.experiment_id,'batch':a.batch,'steps':a.steps,'warmup':a.warmup,
            'precision':'FP32','input_size':640,'seed':SEED,'dataset':'LPI_COCO train',
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__}
    started=time.monotonic()
    try:
        result['phase']='model_setup'
        if yolo:
            from ultralytics import YOLO
            from ultralytics.nn.tasks import DetectionModel
            from ultralytics.utils.torch_utils import ModelEMA
            from tools.lpi_yolo import LpiDetectionTrainer
            # Match the formal trainer's 200-epoch auto-optimizer decision, not tiny smoke AdamW.
            pretrained=YOLO(str(ROOT/'.cache/lpi_pretrained'/c['model'])).model
            model=DetectionModel(pretrained.yaml,nc=5,ch=3,verbose=False)
            model.load(pretrained);del pretrained
            model=model.cuda().float().train()
            from ultralytics.cfg import get_cfg
            args=get_cfg(overrides={k:v for k,v in c.items() if k not in ['name','project','model','data']})
            model.args=args
            model.names=dict(enumerate(['LFM','NLFM','BPSK','BFSK','Frank']))
            proxy=object.__new__(LpiDetectionTrainer);proxy.args=args;proxy.data={"nc":5}
            accumulation=max(round(args.nbs/a.batch),1)
            decay=args.weight_decay*a.batch*accumulation/args.nbs
            import math
            optimizer=proxy.build_optimizer(model,name=args.optimizer,lr=args.lr0,momentum=args.momentum,
                decay=decay,iterations=math.ceil(39600/max(a.batch,args.nbs))*200)
            ema=ModelEMA(model)
            result['optimizer']=type(optimizer).__name__;result['accumulate']=accumulation
            # Profile every step with an optimizer update: conservative relative to accumulation.
            from PIL import Image
            def cpu_batch(ids):
                tensors=[];labels=[];boxes=[];indices=[]
                for j,i in enumerate(ids):
                    im=images[i]
                    with Image.open(DATA/im['file_name']) as image:
                        tensors.append(torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1))
                    for ann in gt[i]:
                        x,y,w,h=ann['bbox'];labels.append([ann['category_id']-1]);boxes.append([(x+w/2)/640,(y+h/2)/640,w/640,h/640]);indices.append(j)
                return {'img':torch.stack(tensors),'cls':torch.tensor(labels,dtype=torch.float32).reshape(-1,1),
                        'bboxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),'batch_idx':torch.tensor(indices,dtype=torch.float32)}
            def run_batch(batch,train=True):
                b={k:v.cuda(non_blocking=False) for k,v in batch.items()};b['img']=b['img'].float()/255
                if not train:
                    with torch.no_grad():
                        val_model=ema.ema
                        predictions=val_model(b['img'])
                        val_loss=val_model.loss(b,predictions)[1]
                        if not torch.isfinite(val_loss).all():raise RuntimeError('nonfinite validation loss')
                    return None
                loss,_=model(b);total=loss.sum();total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                optimizer.step();ema.update(model)
                return total.detach()
        else:
            from detectron2.config import get_cfg
            from detectron2.modeling import build_model
            from diffusiondet import add_diffusiondet_config,add_mobilenetv4_config
            from diffusiondet.dataset_mapper import DiffusionDetDatasetMapper
            from diffusiondet.util.model_ema import add_model_ema_configs
            from train_net import Trainer
            cfg=get_cfg();add_diffusiondet_config(cfg);add_mobilenetv4_config(cfg);add_model_ema_configs(cfg)
            cfg.merge_from_file(str(config));cfg.SOLVER.IMS_PER_BATCH=a.batch
            cfg.SOLVER.BASE_LR=.000035*a.batch/25
            model=build_model(cfg).train();optimizer=Trainer.build_optimizer(cfg,model)
            result['optimizer']=type(optimizer).__name__;result['accumulate']=1
            mapper=DiffusionDetDatasetMapper(cfg,is_train=True)
            def cpu_batch(ids):
                return [mapper({**images[i],'file_name':str(DATA/images[i]['file_name']),
                    'annotations':[{**ann,'bbox_mode':1,'category_id':ann['category_id']-1} for ann in gt[i]]}) for i in ids]
            def run_batch(batch,train=True):
                if not train:
                    with torch.no_grad():model(batch)
                    return None
                loss=model(batch);total=sum(loss.values());total.backward();optimizer.step()
                return total.detach()
        # Three distinct real batches plus a maximum-target-count stress batch.
        batches=[cpu_batch(sample_ids(images,gt,a.batch,k)) for k in range(3)]
        stress=cpu_batch(sample_ids(images,gt,a.batch,3,True))
        baseline_mem=torch.cuda.memory_allocated()/1024**2
        torch.cuda.reset_peak_memory_stats()
        times=[];losses=[]
        result['phase']='training'
        for step in range(a.warmup+a.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();start=time.monotonic()
            total=run_batch(batches[step%3])
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            if not torch.isfinite(total):raise RuntimeError('nonfinite loss')
            losses.append(float(total))
            if step>=a.warmup:times.append(elapsed)
        result['phase']='five_target_stress'
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            total=run_batch(stress)
            if not torch.isfinite(total):raise RuntimeError('nonfinite stress loss')
        if not yolo:
            result['phase']='all_noise_stress'
            noise_ids=sorted(i for i in images if not gt[i])[:a.batch]
            noise_batch=cpu_batch(noise_ids)
            optimizer.zero_grad(set_to_none=True)
            total=run_batch(noise_batch)
            if not torch.isfinite(total):raise RuntimeError('nonfinite all-noise loss')
            result['all_noise_loss']=float(total)
        if not all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):
            raise RuntimeError('nonfinite gradients')
        optimizer.zero_grad(set_to_none=True)
        # Native YOLO uses 2x batch for periodic validation; include that capacity requirement.
        result['phase']='validation'
        model.eval()
        if yolo:
            # Match native training's pre-validation allocator cache release.
            torch.cuda.empty_cache()
        validation=cpu_batch(sample_ids(images,gt,a.batch*2,7,True)) if yolo else cpu_batch(sample_ids(images,gt,1,7,True))
        run_batch(validation,False);torch.cuda.synchronize()
        result.update(status='completed',finite_gradients=True,loss_first=losses[0],loss_last=losses[-1],
            median_step_seconds=statistics.median(times),mean_step_seconds=statistics.mean(times),
            images_per_second=a.batch/statistics.mean(times),step_seconds=times,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,baseline_allocated_mib=baseline_mem,
            validation_batch=a.batch*2 if yolo else 1,stress_targets_per_image=5)
    except BaseException as error:
        result.update(status='oom' if isinstance(error,torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower() else 'failed',
                      error=f'{type(error).__name__}: {error}',peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        import traceback
        traceback.print_exc()
    finally:
        done.set();thread.join(timeout=5)
        result['elapsed_seconds']=time.monotonic()-started
        result['gpu_peak_used_mib']=max((r[1] for r in gpu_samples),default=None)
        result['gpu_peak_utilization']=max((r[2] for r in gpu_samples),default=None)
        result['gpu_samples']=gpu_samples
        write_json(a.output/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['gpu_samples','step_seconds']},ensure_ascii=False),flush=True)
    if result['status'] not in ['completed','oom']:raise SystemExit(1)

if __name__=='__main__':main()
