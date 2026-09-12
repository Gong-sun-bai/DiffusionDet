"""Deterministic single-image LPI inference in the model's own environment."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.lpi_common import (SEED, PROTOCOL, NAMES, load_annotations, prediction_identity,
                              write_json, sha256, resolved_config)


def diffusion_config(path, weights):
    from detectron2.config import get_cfg
    from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config
    from diffusiondet.util.model_ema import add_model_ema_configs
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(str(path))
    cfg.MODEL.WEIGHTS = str(weights)
    # A full trained checkpoint supplies all parameters; inference must not download ImageNet again.
    cfg.MODEL.TIMM.PRETRAINED = False
    cfg.MODEL.DEVICE = 'cuda'
    cfg.freeze()
    return cfg


def predict(config, weights, data_root, split, output):
    import torch
    from tools.lpi_train import check_environment
    check_environment('model' in resolved_config(Path(config)))
    if not torch.cuda.is_available():
        raise RuntimeError('LPI 推理要求 CUDA 可用')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _, images, _ = load_annotations(data_root, split)
    identity = prediction_identity(config, weights, data_root, split)
    cfg = resolved_config(Path(config))
    yolo = 'model' in cfg
    env = {'torch': torch.__version__, 'python': sys.version, 'device': torch.cuda.get_device_name(0)}
    if yolo:
        from ultralytics import YOLO
        env['ultralytics'] = importlib.metadata.version('ultralytics')
        model = YOLO(str(weights))
        if [model.names[i] for i in range(5)] != NAMES or len(model.names) != 5:
            raise ValueError('checkpoint 类别映射不是 LPI 五类')
    else:
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.data import detection_utils as utils
        from diffusiondet.dataset_mapper import DiffusionDetDatasetMapper
        from detectron2.utils.env import seed_all_rng
        dcfg = diffusion_config(config, weights)
        model = build_model(dcfg)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
        expected = model.state_dict()
        actual = checkpoint['model']
        if set(actual) != set(expected) or any(actual[k].shape != expected[k].shape for k in expected):
            raise ValueError('DiffusionDet checkpoint 参数名或 shape 不完整匹配，拒绝部分加载')
        del checkpoint, actual, expected
        DetectionCheckpointer(model).load(str(weights))
        model.eval()
        mapper = DiffusionDetDatasetMapper(dcfg, is_train=False)
        env['timm'] = importlib.metadata.version('timm')
    parameter_count = sum(p.numel() for p in (model.model if yolo else model).parameters())
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prediction_count = 0
    temporary = output / 'predictions.jsonl.partial'
    write_json(output / 'manifest.json', {'status': 'running', 'identity': identity, 'environment': env})
    with temporary.open('w', encoding='utf-8') as f, torch.inference_mode():
        for n, i in enumerate(sorted(images), 1):
            im = images[i]
            derived_seed = int.from_bytes(hashlib.sha256(f'{SEED}:{split}:{i}'.encode()).digest()[:4], 'big')
            if yolo:
                torch.manual_seed(derived_seed)
                result = model.predict(str(Path(data_root)/im['file_name']), imgsz=640, batch=1, device=0,
                                       conf=0.0, iou=.7, max_det=100, agnostic_nms=False,
                                       augment=False, quantize=32, rect=False, verbose=False, save=False)[0]
                xyxy = result.boxes.xyxy.cpu().tolist()
                scores = result.boxes.conf.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
            else:
                seed_all_rng(derived_seed)
                item = mapper({**im, 'image_id': i, 'file_name': str(Path(data_root)/im['file_name'])})
                result = model([item])[0]['instances'].to('cpu')
                xyxy = result.pred_boxes.tensor.tolist()
                scores = result.scores.tolist()
                classes = result.pred_classes.tolist()
            ordered = sorted(range(len(scores)), key=lambda k: -scores[k])[:100]
            detections = []
            for k in ordered:
                x1,y1,x2,y2 = xyxy[k]
                detections.append({'category_id': int(classes[k])+1, 'score': float(scores[k]),
                                   'bbox': [x1,y1,max(0.,x2-x1),max(0.,y2-y1)]})
            f.write(json.dumps({'image_id': i, 'detections': detections}, allow_nan=False, separators=(',',':'))+'\n')
            prediction_count += len(detections)
            if n % 500 == 0:
                print(f'{split}: {n}/{len(images)}, elapsed={time.monotonic()-started:.1f}s', flush=True)
    temporary.rename(output/'predictions.jsonl')
    write_json(output/'manifest.json', {'status': 'completed', 'identity': identity, 'environment': env,
               'image_count': len(images), 'image_ids_sha256': hashlib.sha256(json.dumps(sorted(images)).encode()).hexdigest(),
               'parameter_count_before_inference_fusion': parameter_count, 'prediction_count': prediction_count, 'predictions_sha256': sha256(output/'predictions.jsonl'),
               'peak_memory_mb': torch.cuda.max_memory_allocated()/1024**2, 'elapsed_seconds': time.monotonic()-started})
    print(f'预测完成：{output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    predict(a.config,a.weights,a.data_root,a.split,a.output)

if __name__=='__main__':
    main()
