# DiffusionDet + MobileNetV4 集成指南

本文档介绍如何在 DiffusionDet 项目中使用 MobileNetV4 作为骨干网络进行目标检测训练。

---

## 📁 项目架构概览

```
DiffusionDet/
├── configs/                              # 配置文件目录
│   ├── Base-DiffusionDet.yaml           # 基础配置（公共参数）
│   ├── diffdet.SAR_coco.res50.yaml      # ResNet-50 版本配置
│   ├── diffdet.SAR_coco.mobilenetv4_small.yaml   # 【新增】MobileNetV4-Small 配置
│   └── diffdet.SAR_coco.mobilenetv4_medium.yaml  # 【新增】MobileNetV4-Medium 配置
├── diffusiondet/                         # DiffusionDet 核心模块
│   ├── __init__.py                       # 模块导出
│   ├── detector.py                       # DiffusionDet 检测器主类
│   ├── head.py                           # 检测头 (DynamicHead)
│   ├── config.py                         # DiffusionDet 配置项
│   ├── loss.py                           # 损失函数
│   ├── swintransformer.py               # Swin Transformer 骨干网络
│   └── mobilenetv4.py                   # 【新增】MobileNetV4 骨干网络
├── detectron2/                           # Detectron2 框架
│   ├── modeling/
│   │   └── backbone/                     # 骨干网络模块
│   │       ├── build.py                  # BACKBONE_REGISTRY 注册表
│   │       ├── fpn.py                    # FPN 特征金字塔
│   │       └── resnet.py                 # ResNet 骨干网络
│   └── ...
├── SAR_COCO/                             # SAR 船舶数据集
│   ├── annotations/                      # COCO 格式标注
│   ├── train2017/                        # 训练图像
│   └── val2017/                          # 验证图像
├── train_net.py                          # 训练入口脚本
└── README_MobileNetV4.md                # 本文档
```

---

## 🔄 骨干网络工作流程

DiffusionDet 的骨干网络加载流程如下：

```
配置文件 (cfg.MODEL.BACKBONE.NAME)
        ↓
BACKBONE_REGISTRY 注册表查找
        ↓
build_backbone(cfg, input_shape)
        ↓
骨干网络 (MobileNetV4/ResNet/Swin)
        ↓
FPN 特征金字塔融合
        ↓
输出多尺度特征 {"p2", "p3", "p4", "p5"}
        ↓
DynamicHead 检测头
        ↓
检测结果
```

### 关键代码路径

1. **配置加载**：`train_net.py` → `setup()` → `add_mobilenetv4_config(cfg)`
2. **模型构建**：`detector.py` → `build_backbone(cfg)` → `BACKBONE_REGISTRY.get(name)`
3. **特征提取**：`MobileNetV4.forward()` → `FPN.forward()` → `DynamicHead.forward()`

---

## 🏗️ MobileNetV4 架构

MobileNetV4 是 Google 在 2024 年提出的最新移动端骨干网络，具有以下特点：

### 核心模块

| 模块 | 说明 |
|------|------|
| **UIB** (Universal Inverted Bottleneck) | 统一的倒残差模块，支持多种配置 |
| **Fused IB** | 融合的倒残差模块，用于浅层特征 |
| **ExtraDW** | 额外的深度可分离卷积配置 |

### 模型规格对比

| 模型 | Stage1 | Stage2 | Stage3 | Stage4 | 参数量 | 适用场景 |
|------|--------|--------|--------|--------|--------|----------|
| MobileNetV4-Small | 32 | 64 | 96 | 128 | ~3.8M | 边缘设备 |
| MobileNetV4-Medium | 48 | 80 | 160 | 256 | ~9.7M | 移动端 |
| MobileNetV4-Large | 48 | 96 | 192 | 512 | ~32M | 服务器 |
| ResNet-18 | 64 | 128 | 256 | 512 | ~11.7M | 通用 |
| ResNet-50 | 256 | 512 | 1024 | 2048 | ~25.6M | 高精度 |

---

## ⚙️ 配置说明

### MobileNetV4 配置项

```yaml
MODEL:
  # 骨干网络选择
  BACKBONE:
    NAME: "build_mobilenetv4_fpn_backbone"  # 使用 MobileNetV4 + FPN
    FREEZE_AT: 0  # 冻结的阶段数 (0=不冻结)
  
  # MobileNetV4 专属配置
  MOBILENETV4:
    SIZE: "small"           # 模型大小: "small", "medium", "large"
    OUT_FEATURES: ["stage1", "stage2", "stage3", "stage4"]  # 输出特征层
    NORM: "BN"              # 归一化类型: "BN", "GN"
    WIDTH_MULTIPLIER: 1.0   # 宽度乘数 (0.5~2.0)
  
  # FPN 配置（需与骨干网络匹配）
  FPN:
    IN_FEATURES: ["stage1", "stage2", "stage3", "stage4"]  # 输入特征名
    OUT_CHANNELS: 64        # FPN 输出通道数
```

### 完整配置示例

参考 `configs/diffdet.SAR_coco.mobilenetv4_small.yaml`：

```yaml
_BASE_: "Base-DiffusionDet.yaml"

MODEL:
  BACKBONE:
    NAME: "build_mobilenetv4_fpn_backbone"
  
  MOBILENETV4:
    SIZE: "small"
    OUT_FEATURES: ["stage1", "stage2", "stage3", "stage4"]
    NORM: "BN"
    WIDTH_MULTIPLIER: 1.0
  
  FPN:
    IN_FEATURES: ["stage1", "stage2", "stage3", "stage4"]
    OUT_CHANNELS: 64
  
  ROI_HEADS:
    IN_FEATURES: ["p2", "p3", "p4", "p5"]
  
  DiffusionDet:
    NUM_PROPOSALS: 500
    NUM_CLASSES: 1
    HIDDEN_DIM: 64

DATASETS:
  TRAIN: ("sar_ship_train",)
  TEST: ("sar_ship_val",)

SOLVER:
  MAX_ITER: 99400
  CHECKPOINT_PERIOD: 5000
  STEPS: (69580, 89514)

INPUT:
  CROP:
    ENABLED: True

OUTPUT_DIR: "./output_mobilenetv4_small"
```

---

## 🚀 使用方法

### 1. 训练模型

**使用 MobileNetV4-Small：**
```bash
python train_net.py --num-gpus 1 \
    --config-file configs/diffdet.SAR_coco.mobilenetv4_small.yaml
```

**使用 MobileNetV4-Medium：**
```bash
python train_net.py --num-gpus 1 \
    --config-file configs/diffdet.SAR_coco.mobilenetv4_medium.yaml
```

**多 GPU 训练：**
```bash
python train_net.py --num-gpus 4 \
    --config-file configs/diffdet.SAR_coco.mobilenetv4_small.yaml
```

### 2. 恢复训练

```bash
python train_net.py --num-gpus 1 \
    --config-file configs/diffdet.SAR_coco.mobilenetv4_small.yaml \
    --resume
```

### 3. 模型评估

```bash
python train_net.py --num-gpus 1 \
    --config-file configs/diffdet.SAR_coco.mobilenetv4_small.yaml \
    --eval-only \
    MODEL.WEIGHTS output_mobilenetv4_small/model_final.pth
```

### 4. 命令行覆盖配置

```bash
# 调整学习率和批大小
python train_net.py --num-gpus 1 \
    --config-file configs/diffdet.SAR_coco.mobilenetv4_small.yaml \
    SOLVER.BASE_LR 0.00005 \
    SOLVER.IMS_PER_BATCH 32

# 使用不同的宽度乘数
python train_net.py --num-gpus 1 \
    --config-file configs/diffdet.SAR_coco.mobilenetv4_small.yaml \
    MODEL.MOBILENETV4.WIDTH_MULTIPLIER 0.75
```

---

## 📝 新增文件说明

### 1. `diffusiondet/mobilenetv4.py`

MobileNetV4 骨干网络实现，包含以下组件：

| 类/函数 | 说明 |
|---------|------|
| `ConvBNAct` | 卷积 + BatchNorm + 激活函数组合 |
| `SqueezeExcite` | SE 注意力模块 |
| `UniversalInvertedBottleneck` | UIB 核心模块 |
| `MobileNetV4` | 主干网络类（继承 Backbone） |
| `build_mobilenetv4_backbone` | 构建纯 MobileNetV4 |
| `build_mobilenetv4_fpn_backbone` | 构建 MobileNetV4 + FPN |
| `add_mobilenetv4_config` | 添加配置项到 cfg |

### 2. `configs/diffdet.SAR_coco.mobilenetv4_small.yaml`

MobileNetV4-Small 版本配置，适合资源受限场景。

### 3. `configs/diffdet.SAR_coco.mobilenetv4_medium.yaml`

MobileNetV4-Medium 版本配置，平衡精度与速度。

---

## 🔧 修改的现有文件

### 1. `diffusiondet/__init__.py`

添加 MobileNetV4 模块导出：

```python
from .mobilenetv4 import (
    MobileNetV4,
    build_mobilenetv4_backbone,
    build_mobilenetv4_fpn_backbone,
    add_mobilenetv4_config,
)
```

### 2. `train_net.py`

在 `setup()` 函数中添加配置注册：

```python
from diffusiondet import ..., add_mobilenetv4_config

def setup(args):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)  # 新增
    add_model_ema_configs(cfg)
    ...
```

---

## 💡 调参建议

### 针对 SAR 船舶检测

| 参数 | Small 推荐值 | Medium 推荐值 | 说明 |
|------|-------------|---------------|------|
| `FPN.OUT_CHANNELS` | 64 | 96 | FPN 输出通道 |
| `HIDDEN_DIM` | 64 | 96 | Transformer 隐藏层 |
| `NUM_PROPOSALS` | 300~500 | 300~500 | 候选框数量 |
| `BASE_LR` | 2.5e-5 | 2.5e-5 | 学习率 |
| `IMS_PER_BATCH` | 32~64 | 16~32 | 批大小 |
| `WIDTH_MULTIPLIER` | 1.0 | 1.0 | 宽度乘数 |

### 进一步优化方向

1. **预训练权重**：可加载 ImageNet 预训练的 MobileNetV4 权重
2. **宽度乘数**：使用 `WIDTH_MULTIPLIER: 0.75` 进一步压缩模型
3. **知识蒸馏**：使用 ResNet-50 版本作为教师模型

---

## 📚 参考资料

- [MobileNetV4 论文](https://arxiv.org/abs/2404.10518) - "MobileNetV4: Universal Models for the Mobile Ecosystem"
- [DiffusionDet 论文](https://arxiv.org/abs/2211.09788) - "DiffusionDet: Diffusion Model for Object Detection"
- [Detectron2 文档](https://detectron2.readthedocs.io/)

---

## 📋 更新日志

| 日期 | 内容 |
|------|------|
| 2026-02-05 | 初始版本，添加 MobileNetV4 骨干网络支持 |

---

## ⚠️ 注意事项

1. **特征层名称**：MobileNetV4 使用 `stage1~stage4`，与 ResNet 的 `res2~res5` 不同
2. **通道数匹配**：FPN 的 `OUT_CHANNELS` 需根据骨干网络调整
3. **冻结策略**：MobileNetV4 较轻量，通常不需要冻结早期层
4. **内存占用**：MobileNetV4-Small 显存占用约为 ResNet-50 的 1/3
