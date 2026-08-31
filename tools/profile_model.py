#!/usr/bin/env python3
"""Profile a DiffusionDet configuration without creating an experiment run."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model

from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config
from diffusiondet.datasets import register_project_datasets
from diffusiondet.dataset_mapper import DiffusionDetDatasetMapper
from diffusiondet.util.model_ema import add_model_ema_configs


def load_cfg(path: Path, options, device: str):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(str(path))
    cfg.merge_from_list(list(options))
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def synchronize(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def profile(args):
    register_project_datasets(ROOT)
    cfg = load_cfg(args.config.resolve(), args.opts, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg)
    if args.weights:
        DetectionCheckpointer(model).resume_or_load(str(args.weights.resolve()), resume=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = sum(p.numel() for p in model.parameters())
    state_elements = sum(value.numel() for value in model.state_dict().values())
    model.eval()

    image = torch.zeros(3, args.image_size, args.image_size, device=args.device)
    batched = [
        {"image": image, "height": args.image_size, "width": args.image_size}
        for _ in range(args.batch_size)
    ]
    with torch.no_grad():
        normalized = model.normalizer(image)
        backbone_input = normalized[None].repeat(args.batch_size, 1, 1, 1)
        feature_values = model.backbone(backbone_input)
        feature_shapes = {name: list(value.shape) for name, value in feature_values.items()}
        for _ in range(args.warmup):
            model(batched)
        synchronize(args.device)
        durations = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            model(batched)
            synchronize(args.device)
            durations.append(time.perf_counter() - started)
    inference_peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**2
        if args.device.startswith("cuda")
        else None
    )

    train_step = None
    if args.train_step:
        train_cfg = cfg.clone()
        train_cfg.defrost()
        train_cfg.DATALOADER.NUM_WORKERS = 0
        train_cfg.freeze()
        loader = build_detection_train_loader(
            train_cfg,
            mapper=DiffusionDetDatasetMapper(train_cfg, is_train=True),
        )
        batch = next(iter(loader))
        model.train()
        model.zero_grad(set_to_none=True)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        synchronize(args.device)
        started = time.perf_counter()
        losses = model(batch)
        total_loss = sum(losses.values())
        total_loss.backward()
        synchronize(args.device)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        train_step = {
            "seconds": time.perf_counter() - started,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
                if args.device.startswith("cuda")
                else None
            ),
            "losses": {key: float(value.detach()) for key, value in losses.items()},
            "all_gradients_finite": bool(gradients)
            and all(bool(torch.isfinite(value).all()) for value in gradients),
        }
        model.zero_grad(set_to_none=True)

    pretrained_source = None
    if cfg.MODEL.BACKBONE.NAME == "build_timm_fpn_backbone":
        timm_model = model.backbone.bottom_up.bottom_up
        pretrained_cfg = getattr(timm_model, "pretrained_cfg", {})
        pretrained_source = {
            key: pretrained_cfg.get(key)
            for key in ("url", "hf_hub_id", "architecture", "tag")
            if pretrained_cfg.get(key)
        }

    result = {
        "config": str(args.config.resolve().relative_to(ROOT)),
        "device": args.device,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "state_dict_elements": state_elements,
        "below_9_5m": trainable < 9_500_000,
        "feature_shapes": feature_shapes,
        "latency_seconds_median": statistics.median(durations),
        "latency_seconds_min": min(durations),
        "batch_size": args.batch_size,
        "peak_memory_mib": inference_peak_memory,
        "train_step": train_step,
        "timm": {
            "name": str(cfg.MODEL.TIMM.NAME),
            "pretrained": bool(cfg.MODEL.TIMM.PRETRAINED),
            "out_indices": list(cfg.MODEL.TIMM.OUT_INDICES),
            "pretrained_source": pretrained_source,
        }
        if cfg.MODEL.BACKBONE.NAME == "build_timm_fpn_backbone"
        else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--train-step",
        action="store_true",
        help="Run one real dataset forward/backward batch and report gradients/memory.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    result = profile(args)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
