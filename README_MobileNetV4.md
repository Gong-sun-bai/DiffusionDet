# DiffusionDet + MobileNetV4 集成说明

本文说明本仓库 MobileNetV4 骨干的实现位置、历史结论和后续实验方式。统一实验规范、完整指标与限制见 `docs/实验日志.md`。

## 代码与配置

关键文件：

```text
diffusiondet/mobilenetv4.py
diffusiondet/__init__.py
train_net.py
configs/experiments/sar_ship/sar-003.yaml
configs/experiments/sar_ship/sar-004.yaml
```

调用链：

```text
MODEL.BACKBONE.NAME
  -> BACKBONE_REGISTRY
  -> build_mobilenetv4_fpn_backbone()
  -> MobileNetV4 多尺度特征
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

## 历史结果

| ID | 骨干 | 完整检测模型参数 | AP | AP50 | AP75 | 已知限制 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sar-003` | MobileNetV4-Small | 7.70M | 54.96 | 94.16 | 57.76 | `SEED=-1`，无端侧测速 |
| `sar-004` | MobileNetV4-Medium | 20.58M | 57.61 | 95.12 | 63.26 | 延长训练后 milestones 未重算，多次恢复 |

历史结果目录：

```text
runs/sar_ship/sar-003__mnv4-small-fpn64-h64-bs64-it99400-scratch/
runs/sar_ship/sar-004__mnv4-medium-fpn96-h96-bs50-it119280-scratch/
```

Small 相对历史 R50 参考 `sar-000` 参数减少约 93%，AP 下降 9.57；Medium 相对 Small 增加 2.65 AP，但参数约为 2.67 倍。当前没有端到端延迟、吞吐和部署端测试，不能仅凭参数量断言推理更快。

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

## 后续 MobileNetV4 消融规范

开始新的 MobileNetV4 实验前：

1. 先完成 `sar-005` 固定种子 R18 基线；
2. 从最新基线配置继承并分配未使用的新实验 ID；
3. 一次只改一个概念因素；
4. 固定 `SEED=40244023`；
5. 保持所有 `SOLVER.STEPS < SOLVER.MAX_ITER`；
6. 在 `EXPERIMENT.CHANGES` 中写明 `配置键: 旧值 -> 新值`。

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

恢复 `Difdet` 环境后可使用：

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
