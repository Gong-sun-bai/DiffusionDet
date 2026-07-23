"""统计指定 DiffusionDet 配置和权重的模型参数量。"""

import argparse
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config
from diffusiondet.util.model_ema import add_model_ema_configs
from experiment_manager import validate_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="例如 cpu 或 cuda:0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_weights(str(args.weights))
    if not args.config_file.is_file():
        raise FileNotFoundError(f"配置不存在：{args.config_file}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"权重不存在：{args.weights}")

    device = torch.device(args.device)
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(str(args.config_file))
    cfg.MODEL.WEIGHTS = str(args.weights)
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()

    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Device: {device}")
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")
    print("\nParameters per top-level module:")
    for name, module in model.named_children():
        module_total = sum(parameter.numel() for parameter in module.parameters())
        module_trainable = sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        print(
            f"{name:20s} | Total: {module_total / 1e6:.2f} M | "
            f"Trainable: {module_trainable / 1e6:.2f} M"
        )


if __name__ == "__main__":
    main()
