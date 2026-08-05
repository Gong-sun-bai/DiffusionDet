# Codex 项目上下文

本文件适用于整个仓库，是 Codex 重新进入项目时的第一入口。默认使用简体中文协作，命令、路径、配置键和代码标识符保留原文。

## 开始工作前

依次阅读：

1. `AGENTS.md`
2. `docs/项目交接与重启指南.md`
3. `docs/实验日志.md`
4. `docs/RTX2080Ti基线复现记录.md`

先执行只读检查：

```bash
git status --short --branch
python3 tools/validate_experiment_configs.py
python3 tools/migrate_historical_runs.py --verify
```

不要默认当前 shell 已激活训练环境。系统 Python 缺 PyTorch，`yolo` 也不是项目环境；当前可用环境是 `/home/troy/anaconda3/envs/Difdet`。激活后必须验证：

```bash
conda activate Difdet
python -c "import torch, torchvision, fvcore, pycocotools, timm; print(torch.__version__, torch.cuda.is_available())"
python -c "import detectron2; from detectron2 import _C; print(detectron2.__file__, _C.__file__)"
```

`Difdet` 已于 2026-07-23 完成 Batch 16、20 iter 的 `sar-test-001` 冒烟训练；当前 `_C` 为 CPU-only，普通 ROIAlign/NMS 由 TorchVision CUDA 实现。环境重建与限制见 `docs/RTX2080Ti基线复现记录.md`。

## 项目定位

本仓库是 DiffusionDet 的本地研究分支，主要研究：

- SAR 船舶单类别检测；
- PANDA 四类别检测；
- ResNet-50、ResNet-18 与 MobileNetV4 的精度/体量权衡；
- 固定基线后的单因素模型与训练策略消融。

核心入口：

| 路径 | 用途 |
| --- | --- |
| `train_net.py` | 训练、恢复、评估 |
| `experiment_manager.py` | 实验元数据校验、目录保护、清单生命周期 |
| `diffusiondet/datasets.py` | SAR/PANDA 集中注册 |
| `configs/experiments/` | 本地历史、基线与消融配置 |
| `docs/实验日志.md` | 结果、假设、限制和后续队列的唯一人工日志 |
| `runs/` | 本地结果和权重；被 Git 忽略 |

上游 COCO/LVIS 配置保留在 `configs/` 根目录，不应为了本地实验批量改写。

## 数据集

实际本地目录：

- `SAR_COCO_ship/`：31,783 张训练图、7,946 张验证图，单类 `ship`；
- `panda_coco_data/`：3,925 张训练图、562 张验证图，四类别。

`diffusiondet/datasets.py` 会基于仓库根目录同时注册：

- `sar_ship_train` / `sar_ship_val`
- `panda_yolo_train` / `panda_yolo_val`

禁止再靠注释注册语句切换数据集。数据集和权重均不提交 Git。

## 实验目录与 ID

目录由 `EXPERIMENT.*` 自动解析：

```text
runs/<DATASET>/<ID>__<NAME>/
```

当前 ID：

- `sar-000`：历史 R50/256；
- `sar-001`：历史 R18/128，AP 66.94，`SEED=-1`；
- `sar-002`：历史 R18/64，调度无效；
- `sar-003`：历史 MobileNetV4-Small；
- `sar-004`：历史 MobileNetV4-Medium；
- `sar-005`：已完成的固定种子、多尺度 Batch 25 基线，AP 58.84；目录名与实际 Batch/Iter 不一致，以冻结配置为准；
- `sar-006`：已完成的固定 256 可重复基线，最佳 AP 67.77，沿用 `sar-005` 的 Batch 25 和 254264 iter；
- `sar-test-001`：已完成的 RTX 2080 Ti Batch 16 / 20 iter 冒烟，不含 AP；
- `panda-000`：历史 PANDA 四类别基线。

下一未占用正式 SAR 序号为 `sar-007`，下一测试序号为 `sar-test-002`。正式 ID 使用 `<dataset>-NNN`，测试实验直接使用 `<dataset>-test-NNN`；测试 ID 只用于 smoke，不进入正式实验排名。NAME 只允许小写字母、数字和连字符，不得包含 AP 或结论。

历史迁移映射只在 `docs/实验日志.md` 和 `tools/migrate_historical_runs.py` 中维护；不要恢复旧 `output*` 目录或创建兼容软链接。

## 运行安全规则

- 新训练发现目标目录非空会报错，不要绕过。
- `--resume` 只允许在已有规范目录中使用，且 `last_checkpoint` 必须指向现有检查点。
- `--eval-only` 不能与 `--resume` 同时使用。
- 评估必须通过命令行显式指定 `MODEL.WEIGHTS`，文件名只允许 `model_final.pth`、`model_latest.pth` 或 `model_best.pth`。
- `inference/instances_predictions.pth` 是逐图预测列表，永远不是模型权重。
- 评估输出自动进入 `<run>/evaluations/<UTC时间戳>/`。
- 所有训练和 `--resume` 都必须固定非负 `SEED`，默认 `40244023`；评估旧配置时不检查训练种子。
- 所有训练和 `--resume` 的每个 `SOLVER.STEPS` 必须小于 `SOLVER.MAX_ITER`；评估旧配置时允许保留历史调度错误。
- `SOLVER.CHECKPOINT_RETENTION=latest` 会周期性覆盖 `model_latest.pth`；启用 `TEST.BEST_CHECKPOINT` 时按验证指标覆盖 `model_best.pth`，`last_checkpoint` 仍必须指向 latest 以用于续训。
- 不修改历史 `config.yaml`、`log.txt`、`metrics.json` 和权重；运行清单使用 schema v2 的 `description` 汇总说明，新结论写入实验日志。

新实验训练（先创建并校验新的 `sar-007` 配置）：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/<sar-007-new-experiment>.yaml
```

标准复评：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-001.yaml \
  --eval-only \
  MODEL.WEIGHTS runs/sar_ship/sar-001__r18-fpn128-h128-bs16-it397288-scratch/model_final.pth
```

## 消融规范

新消融配置使用可选的统一说明字段：

```yaml
EXPERIMENT:
  ID: "sar-XYZ"
  NAME: "..."
  DATASET: "sar_ship"
  DESCRIPTION: "以 sar-006 为对照，将 NUM_PROPOSALS 从 500 改为 300，用于检验减少 proposal 对精度和开销的影响。"
  OUTPUT_ROOT: "./runs"
```

`DESCRIPTION` 是不参与运行逻辑的可选自由文本，建议简要写明对照实验、主要改动、目的和重要限制。一次实验只验证一个假设；FPN/HIDDEN 等因结构兼容必须一起变化时，可作为一个“检测头宽度”概念因素，并在 `DESCRIPTION` 和日志中说明。不要在模型源码中按实验 ID 写条件分支；应新增语义清晰的配置开关并保持默认行为不变。

推荐顺序：以 `sar-006` 为可信固定 256 基线 → 通道宽度 → backbone → proposal 数 → sample step → 输入尺度/裁剪 → 损失权重。

## 历史结果速查

| ID | 数据集 | 参数量 | AP | 关键限制 |
| --- | --- | ---: | ---: | --- |
| `sar-000` | SAR | 110.67M | 64.53 | 冻结配置的权重字段曾被失败评估覆盖 |
| `sar-001` | SAR | 34.49M | 66.94 | `SEED=-1` |
| `sar-002` | SAR | 17.88M | 60.96 | `STEPS > MAX_ITER`，状态 `invalid` |
| `sar-003` | SAR | 7.70M | 54.96 | 无端侧速度证据 |
| `sar-004` | SAR | 20.58M | 57.61 | 延长训练后 milestones 未重算 |
| `sar-005` | SAR | 34.46M | 58.84 | 固定种子但为 Batch 25、多尺度，不是 `sar-001` 严格复现 |
| `sar-006` | SAR | 34.46M | 67.77 | 固定种子、Batch 25、固定 256；当前可重复基线 |
| `panda-000` | PANDA | 110.67M | 13.24 | 四类别，不可与 SAR 横比 |

最终指标优先读取 `metrics.json` 最后一条含 `bbox/AP` 的记录，并与 `log.txt` 复核。参数量来自实际评估权重的 `model` 状态字典。

## 修改位置

- backbone：`diffusiondet/mobilenetv4.py`、`detectron2/modeling/backbone/resnet.py`
- FPN：`detectron2/modeling/backbone/fpn.py`
- 检测器/采样：`diffusiondet/detector.py`
- 动态头：`diffusiondet/head.py`
- 损失：`diffusiondet/loss.py`
- 数据映射和增强：`diffusiondet/dataset_mapper.py`
- 实验安全规则：`experiment_manager.py`
- 数据注册：`diffusiondet/datasets.py`

完成实验或修正关键事实后，更新 `docs/实验日志.md`；若影响上手方式，同时更新交接指南和 README。

## 验证层级

静态验证：

```bash
python3 -m unittest tests.test_experiment -v
python3 tools/validate_experiment_configs.py
python3 tools/migrate_historical_runs.py --verify
git diff --check
```

激活 `Difdet` 后按风险递进：

1. 导入配置、注册数据集并查询 Catalog；
2. 构建模型，检查特征名/shape 和 checkpoint 加载；
3. 用全新测试 ID 做短迭代 smoke test；`sar-test-001` 已完成，不得覆盖；
4. 正式训练或时间戳目录复评。

`sar-test-001` 的 R18/FPN128 Batch 16 实测峰值显存为 12,626 MB；`sar-006` 固定 256 Batch 25 的完整训练峰值为 15,620 MB。历史其他结构曾达 24–27 GB。正式训练前仍需检查显存；调整 Batch 时按复现记录同步缩放学习率、MAX_ITER、WARMUP_ITERS 和 STEPS，并使用新实验 ID。
