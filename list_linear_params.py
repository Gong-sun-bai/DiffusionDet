"""列出指定 DiffusionDet 配置中参数量最大的线性层。"""

import argparse
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config
from diffusiondet.util.model_ema import add_model_ema_configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.config_file.is_file():
        raise FileNotFoundError(f"配置不存在：{args.config_file}")

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(str(args.config_file))
    cfg.freeze()

    model = build_model(cfg)
    items = []
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            count = module.weight.numel()
            if module.bias is not None:
                count += module.bias.numel()
            items.append((count, name, tuple(module.weight.shape)))
            total += count

    items.sort(key=lambda item: item[0], reverse=True)
    print(f"Total Linear params: {total / 1e6:.3f}M")
    print("Top Linear layers:")
    for count, name, shape in items[:60]:
        print(f"{count / 1e6:7.3f}M  {name:90s}  W{shape}")


if __name__ == "__main__":
    main()
