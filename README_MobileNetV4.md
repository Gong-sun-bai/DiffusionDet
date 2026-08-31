# DiffusionDet + MobileNetV4 集成说明

本文说明本仓库 MobileNetV4 骨干的实现位置、历史结论和后续实验方式。统一实验规范、完整指标与限制见 `docs/实验日志.md`。

## 代码与配置

关键文件：

```text
diffusiondet/mobilenetv4.py                 # 本地自定义 MobileNetV4
diffusiondet/timm_backbone.py               # 标准 timm features_only 适配器
diffusiondet/__init__.py
train_net.py
configs/experiments/sar_ship/sar-003.yaml
configs/experiments/sar_ship/sar-004.yaml
configs/experiments/sar_ship/sar-011-under10m-timm-mnv4-scratch.yaml
configs/experiments/sar_ship/sar-012-under10m-timm-mnv4-pretrained.yaml
```

调用链：

```text
MODEL.BACKBONE.NAME
  -> BACKBONE_REGISTRY
  -> build_mobilenetv4_fpn_backbone() / build_timm_fpn_backbone()
  -> 四级 stride 4/8/16/32 特征
  -> FPN
  -> DiffusionDet DynamicHead
```

主要配置键：

```yaml
MODEL:
  BACKBONE:
    NAME: "build_mobilenetv4_fpn_backbone"
  MOBILENETV4:
    SIZE: "small"  # small / medium / large
    OUT_FEATURES: ["stage1", "stage2", "stage3", "stage4"]
    NORM: "BN"
    WIDTH_MULTIPLIER: 1.0
  FPN:
    IN_FEATURES: ["stage1", "stage2", "stage3", "stage4"]
    OUT_CHANNELS: 64
  DiffusionDet:
    HIDDEN_DIM: 64
```

FPN 的 `IN_FEATURES` 必须与 MobileNetV4 输出特征名一致；`FPN.OUT_CHANNELS` 与 `DiffusionDet.HIDDEN_DIM` 是结构兼容性耦合参数，应作为一个“检测头宽度”概念因素记录。

标准 `timm` 路径还需配置 `MODEL.TIMM.NAME`、`PRETRAINED`、`OUT_INDICES` 和 `OUT_FEATURES`。适配器会拒绝不是四级 stride 4/8/16/32 的输出。

## 历史结果

| ID | 骨干 | 完整检测模型参数 | AP | AP50 | AP75 | 已知限制 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sar-003` | MobileNetV4-Small | 7.70M | 54.96 | 94.16 | 57.76 | `SEED=-1`，无端侧测速 |
| `sar-004` | MobileNetV4-Medium | 20.58M | 57.61 | 95.12 | 63.26 | 延长训练后 milestones 未重算，多次恢复 |
| `sar-011` | 标准 MobileNetV4-Conv-Small，scratch，共享 H128 | 6.04M | 52.04 | 93.32 | 52.03 | 50k screening |
| `sar-012` | 标准 MobileNetV4-Conv-Small，预训练，共享 H128 | 6.04M | 57.36 | 95.16 | 62.61 | 50k screening，正式候选 |

历史结果目录：

```text
runs/sar_ship/sar-003__mnv4-small-fpn64-h64-bs64-it99400-scratch/
runs/sar_ship/sar-004__mnv4-medium-fpn96-h96-bs50-it119280-scratch/
```

Small 相对历史 R50 参考 `sar-000` 参数减少约 93%，AP 下降 9.57；Medium 相对 Small 增加 2.65 AP，但参数约为 2.67 倍。当前没有端到端延迟、吞吐和部署端测试，不能仅凭参数量断言推理更快。

`sar-012` 与 `sar-011` 只有 ImageNet 预训练开关不同，50k 提高 5.32 AP，证明标准 backbone 预训练是本轮主要有效因素。`sar-012` 的 batch=1 profile 中位延迟约 20.45 ms，但该值只用于同机候选比较，不代表端侧部署性能。其完整训练配置是 `sar-016`。

## 历史权重复评

历史配置被标记为 `historical`，入口禁止重新训练，只能显式指定 `model_final.pth` 评估：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-003.yaml \
  --eval-only \
  MODEL.WEIGHTS runs/sar_ship/sar-003__mnv4-small-fpn64-h64-bs64-it99400-scratch/model_final.pth
```

评估结果自动写入：

```text
runs/sar_ship/sar-003__mnv4-small-fpn64-h64-bs64-it99400-scratch/
└── evaluations/<UTC时间戳>/
```

禁止把 `inference/instances_predictions.pth` 当作权重。

## 当前正式验证方式

不要重跑 `sar-011` 或 `sar-012` screening。提交当前修改后，如需验证 MobileNetV4 正式候选，直接运行其独立配置：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-016-under10m-timm-mnv4-full-seed40244023.yaml
```

`sar-016` 使用种子 `40244023` 从头执行 254,264 iter，保持固定 256、Batch 25、有效 milestones、六级完全共享 H128 头、1-step 和无蒸馏设置。若后续需要更换种子复现，应在分析正式结果后新建未占用实验 ID，不提前创建条件队列。

若正式结果仍不满足目标，后续实验继续遵守：一次只改一个概念因素、固定非负种子、保持所有 `SOLVER.STEPS < SOLVER.MAX_ITER`，并在 `EXPERIMENT.DESCRIPTION` 中记录对照、改动、目的和限制。

建议将下列因素分开验证：

- backbone：R18 与 MobileNetV4 尺寸；
- 检测头宽度：FPN/HIDDEN 的兼容性耦合调整；
- MobileNetV4 `WIDTH_MULTIPLIER`；
- proposal 数；
- diffusion sample step；
- 输入尺度或裁剪；
- 损失权重。

不要用训练 step 时间代替推理性能。轻量化结论至少应同时记录完整检测模型参数量、AP、AP75、峰值显存，并在环境恢复后补充固定输入和固定硬件下的推理延迟/吞吐。

## 参数统计

恢复 `Difdet` 环境后，可对无权重配置执行结构/profile 检查：

```bash
python tools/profile_model.py \
  --config configs/experiments/sar_ship/sar-016-under10m-timm-mnv4-full-seed40244023.yaml \
  --device cuda --batch-size 1 --warmup 3 --iterations 10
```

对已有权重统计参数：

```bash
python param_compare.py \
  --config-file configs/experiments/sar_ship/sar-003.yaml \
  --weights runs/sar_ship/sar-003__mnv4-small-fpn64-h64-bs64-it99400-scratch/model_final.pth
```

查看线性层参数：

```bash
python list_linear_params.py \
  --config-file configs/experiments/sar_ship/sar-003.yaml
```
