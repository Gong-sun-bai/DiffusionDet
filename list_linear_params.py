# list_linear_params.py
import sys
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model

# 关键：注册 DiffusionDet 的自定义 config key（MODEL.DiffusionDet.*）
from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs


def main(cfg_path: str):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)          # ✅ 必须在 merge 之前
    add_model_ema_configs(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    model = build_model(cfg)
    items = []
    total = 0

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            n = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            items.append((n, name, tuple(m.weight.shape)))
            total += n

    items.sort(key=lambda x: x[0], reverse=True)
    print(f"Total Linear params: {total/1e6:.3f}M")
    print("Top Linear layers:")
    for n, name, shape in items[:60]:
        print(f"{n/1e6:7.3f}M  {name:90s}  W{shape}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "output/config.yaml"
    main(cfg_path)
