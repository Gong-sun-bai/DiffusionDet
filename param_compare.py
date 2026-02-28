import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config
from diffusiondet.util.model_ema import add_model_ema_configs
import os

# =========================
# 配置文件和权重路径
# =========================
CONFIG_FILE = "/home/bai/demo/DiffusionDet/output_mobilenetv4_small/config.yaml"
WEIGHTS_FILE = "/home/bai/demo/DiffusionDet/output_mobilenetv4_small/model_final.pth"

# =========================
# 设备选择
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# 加载配置
# =========================
cfg = get_cfg()
add_diffusiondet_config(cfg)
add_mobilenetv4_config(cfg)  # 添加 MobileNetV4 配置
add_model_ema_configs(cfg)
cfg.merge_from_file(CONFIG_FILE)

# 如果 SOLVER.OPTIMIZER 不存在，手动添加
if not hasattr(cfg.SOLVER, "OPTIMIZER"):
    cfg.SOLVER.OPTIMIZER = "ADAMW"

# Override NUM_CLASSES to match the checkpoint (which seems to be 1 class)
cfg.MODEL.DiffusionDet.NUM_CLASSES = 1

cfg.MODEL.WEIGHTS = WEIGHTS_FILE

# =========================
# 构建模型
# =========================
model = build_model(cfg)
model.to(device)
model.eval()  # 不做训练

# =========================
# 加载权重
# =========================
state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location=device)
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict)
print("Weights loaded successfully!")

# =========================
# 统计总参数量和可训练参数量
# =========================
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params / 1e6:.2f} M")
print(f"Trainable params: {trainable_params / 1e6:.2f} M")

# =========================
# 按模块统计参数量
# =========================
print("\nParameters per module:")
for name, module in model.named_children():
    module_params = sum(p.numel() for p in module.parameters())
    module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name:20s} | Total: {module_params / 1e6:.2f} M | Trainable: {module_trainable / 1e6:.2f} M")
