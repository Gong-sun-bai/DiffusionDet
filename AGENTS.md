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

不要默认当前 shell 已激活训练环境。DiffusionDet 训练、评估和统一 COCOeval 使用 `/home/troy/anaconda3/envs/Difdet`；Ultralytics 对比实验单独使用 `/home/troy/anaconda3/envs/yolo`，两个环境不得混用。激活 `Difdet` 后必须验证：

```bash
conda activate Difdet
python -c "import torch, torchvision, fvcore, pycocotools, timm; print(torch.__version__, torch.cuda.is_available())"
python -c "import detectron2; from detectron2 import _C; print(detectron2.__file__, _C.__file__)"
```

运行 YOLO 对比实验前改为验证独立环境：

```bash
conda activate yolo
export PYTHONNOUSERSITE=1
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

当前固定口径为 PyTorch `2.9.1+cu128`、Ultralytics `8.4.91`；`yolo` 环境不用于导入本仓库 Detectron2。

`Difdet` 已于 2026-07-23 完成 Batch 16、20 iter 的 `sar-test-001` 冒烟训练；当前 `_C` 为 CPU-only，普通 ROIAlign/NMS 由 TorchVision CUDA 实现。环境重建与限制见 `docs/RTX2080Ti基线复现记录.md`。

## 项目定位

本仓库是 DiffusionDet 的本地研究分支，主要研究：

- SAR 船舶单类别检测；
- PANDA 四类别检测；
- ResNet-50、ResNet-18、MobileNetV4 与 RepViT 的精度/体量权衡；
- 六级动态头共享、`timm` 轻量 backbone、扩散采样和查询一致蒸馏；
- 固定基线后的单因素模型与训练策略消融。
- YOLO26n、YOLO11n 与 YOLOv8n 的统一预训练 SAR 对比。

核心入口：

| 路径 | 用途 |
| --- | --- |
| `train_net.py` | 训练、恢复、评估 |
| `experiment_manager.py` | 实验元数据校验、目录保护、清单生命周期 |
| `diffusiondet/datasets.py` | SAR/PANDA 集中注册 |
| `configs/experiments/` | 本地历史、基线与消融配置 |
| `configs/yolo/sar_ship/` | SAR YOLO 数据描述与三代 nano 训练配置 |
| `tools/prepare_sar_yolo_dataset.py` | 从原始 SAR COCO 标注幂等生成 YOLO TXT 标签和图片硬链接 |
| `tools/evaluate_yolo_coco.py` | 把 YOLO 预测映射回原始 ID 并执行统一 COCOeval |
| `docs/实验日志.md` | 结果、假设、限制和后续队列的唯一人工日志 |
| `runs/` | 本地结果和权重；被 Git 忽略 |

上游 COCO/LVIS 配置保留在 `configs/` 根目录，不应为了本地实验批量改写。

## 数据集

新增五类雷达时频仿真工具位于 `dataser_process/`，说明与一键生成命令见
`dataser_process/雷达信号仿真数据集说明.md`。默认输出 `LPI_COCO/`，不提交数据。
设计为 99,000 张、LFM/NLFM/BPSK/BFSK/Frank 五类、−10～10 dB 共 11 档 SNR，
train/val/test=39600/9900/49500，含 10% 纯噪声。2026-09-12 正式全量已通过只读核验：
99,000 张图片、237,600 个框，300 张抽样重建哈希一致；已接入五模型配置与分 SNR 评价，正式训练尚未启动。纯噪声不赋 SNR；虚警率按纯噪声观测窗 FP/(FP+TN) 统计，
不能混用错误框数与负图片数。生成器使用 `Difdet` 的 Python 与 CPU，不使用 YOLO 环境。

实际本地目录：

- `SAR_COCO_ship/`：31,783 张训练图、7,946 张验证图，单类 `ship`；
- `panda_coco_data/`：3,925 张训练图、562 张验证图，四类别。

YOLO 训练继续复用 `SAR_COCO_ship/train2017` 与 `val2017` 图片；生成的 TXT 标签、图片硬链接和数据清单位于被 Git 忽略的 `SAR_COCO_ship/yolo/`，不得改写原始 COCO JSON。这里不能使用目录软链接，因为 Ultralytics 会解析真实路径并丢失 `/images/` 标签映射语义。

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
- `sar-007`～`sar-010`：本地 MobileNetV4 与 head sharing 的 20k screening，均已淘汰；
- `sar-011`：标准 MobileNetV4 scratch 50k 对照，已淘汰；
- `sar-012`：标准 MobileNetV4 预训练 50k finalist，AP 57.36、6.04M 参数；
- `sar-013`：RepViT-M0.9 预训练 50k finalist，AP 61.98、9.44M 参数；
- `sar-014`：查询一致蒸馏 50k screening，AP 60.36，已淘汰；
- `sar-015` / `sar-016`：两名 finalist 的固定种子完整训练均已完成，最佳 AP 分别为 67.8786 / 67.2254；
- `panda-000`：历史 PANDA 四类别基线。
- `yolo-sar-001`：计划中的 YOLO26n 预训练统一对比；
- `yolo-sar-002`：计划中的 YOLO11n 预训练统一对比；
- `yolo-sar-003`：计划中的 YOLOv8n 预训练统一对比。

下一未占用正式 DiffusionDet SAR 序号为 `sar-017`，下一测试序号为 `sar-test-002`，下一 YOLO 对比序号为 `yolo-sar-004`。YOLO 使用独立编号，不占用 `sar-NNN`。正式 DiffusionDet ID 使用 `<dataset>-NNN`，测试实验直接使用 `<dataset>-test-NNN`；测试 ID 只用于 smoke，不进入正式实验排名。NAME 只允许小写字母、数字和连字符，不得包含 AP 或结论。

历史迁移映射只在 `docs/实验日志.md` 和 `tools/migrate_historical_runs.py` 中维护；不要恢复旧 `output*` 目录或创建兼容软链接。

## 运行安全规则

- 普通非 resume 训练发现规范目标目录非空时会打印绝对路径并正常退出，不会改写原结果；未完成实验必须显式使用 `--resume`。
- `--resume` 只允许在已有规范目录中使用，且 `last_checkpoint` 必须指向现有检查点。
- `--eval-only` 不能与 `--resume` 同时使用。
- 评估必须通过命令行显式指定 `MODEL.WEIGHTS`，文件名只允许 `model_final.pth`、`model_latest.pth` 或 `model_best.pth`。
- `inference/instances_predictions.pth` 是逐图预测列表，永远不是模型权重。
- 评估输出自动进入 `<run>/evaluations/<UTC时间戳>/`。
- 所有训练和 `--resume` 都必须固定非负 `SEED`，默认 `40244023`；评估旧配置时不检查训练种子。
- 所有训练和 `--resume` 的每个 `SOLVER.STEPS` 必须小于 `SOLVER.MAX_ITER`；评估旧配置时允许保留历史调度错误。
- `SOLVER.CHECKPOINT_RETENTION=latest` 会周期性覆盖 `model_latest.pth`；启用 `TEST.BEST_CHECKPOINT` 时按验证指标覆盖 `model_best.pth`，`last_checkpoint` 仍必须指向 latest 以用于续训。
- 不修改历史 `config.yaml`、`log.txt`、`metrics.json` 和权重；运行清单使用 schema v2 的 `description` 汇总说明，新结论写入实验日志。

`sar-015`、`sar-016` 已完成，不得原地重跑、恢复或延长。SAR/PANDA 正式训练沿用提交后逐个启动的规则；LPI 按用户 2026-09-12 的新要求允许先训练验证再提交，保存源码快照，不使用自动正式训练队列。YOLO 对比须先按 `docs/实验日志.md` 第 5.7 节准备数据，再分别运行：

```bash
yolo detect train cfg=configs/yolo/sar_ship/yolo-sar-001-yolo26n-pretrained.yaml
yolo detect train cfg=configs/yolo/sar_ship/yolo-sar-002-yolo11n-pretrained.yaml
yolo detect train cfg=configs/yolo/sar_ship/yolo-sar-003-yolov8n-pretrained.yaml
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

当前 DiffusionDet 顺序：需要跨种子复现时另建从 `sar-017` 开始的新配置，先复现 `sar-015`，再按资源决定是否复现 `sar-016`。YOLO 对比按 `yolo-sar-001`～`yolo-sar-003` 顺序独立执行。若正式结果仍不满足目标，再以正式最佳结果为基线依次检查 proposal 数、sample step、输入尺度/裁剪和损失权重。

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
| `sar-012` | SAR screening | 6.04M | 57.36 | 50k、单种子、未进入 LR milestones；正式候选 |
| `sar-013` | SAR screening | 9.44M | 61.98 | 50k、单种子、未进入 LR milestones；正式候选 |
| `sar-014` | SAR screening | 9.44M | 60.36 | 蒸馏低于无蒸馏学生，已淘汰 |
| `sar-015` | SAR | 9.44M | 67.88 | 单种子；当前 10M 以下精度优先方案 |
| `sar-016` | SAR | 6.04M | 67.23 | 单种子；当前体量与验证速度优先方案 |
| `panda-000` | PANDA | 110.67M | 13.24 | 四类别，不可与 SAR 横比 |

最终指标优先读取 `metrics.json` 最后一条含 `bbox/AP` 的记录，并与 `log.txt` 复核。参数量来自实际评估权重的 `model` 状态字典。

## 修改位置

- backbone：`diffusiondet/mobilenetv4.py`、`diffusiondet/timm_backbone.py`、`detectron2/modeling/backbone/resnet.py`
- FPN：`detectron2/modeling/backbone/fpn.py`
- 检测器/采样：`diffusiondet/detector.py`
- 动态头：`diffusiondet/head.py`
- 损失：`diffusiondet/loss.py`
- 数据映射和增强：`diffusiondet/dataset_mapper.py`
- 实验安全规则：`experiment_manager.py`
- 分层筛选：`tools/run_under10m_screening.py`、`tools/run_under10m_remaining_screening.py`
- 正式配置：`configs/experiments/sar_ship/sar-015-under10m-repvit-full-seed40244023.yaml`、`sar-016-under10m-timm-mnv4-full-seed40244023.yaml`
- 数据注册：`diffusiondet/datasets.py`
- YOLO 配置与数据桥接：`configs/yolo/sar_ship/`、`tools/prepare_sar_yolo_dataset.py`
- YOLO 统一评估：`tools/evaluate_yolo_coco.py`

完成实验或修正关键事实后，更新 `docs/实验日志.md`；若影响上手方式，同时更新交接指南和 README。

## 验证层级

静态验证：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 tools/validate_experiment_configs.py
python3 tools/migrate_historical_runs.py --verify
python3 tools/prepare_sar_yolo_dataset.py --check-only
git diff --check
```

激活 `Difdet` 后按风险递进：

1. 导入配置、注册数据集并查询 Catalog；
2. 构建模型，检查特征名/shape 和 checkpoint 加载；
3. 用全新测试 ID 做短迭代 smoke test；`sar-test-001` 已完成，不得覆盖；
4. 正式训练或时间戳目录复评。

`sar-test-001` 的 R18/FPN128 Batch 16 实测峰值显存为 12,626 MB；`sar-006` 固定 256 Batch 25 的完整训练峰值为 15,620 MB。历史其他结构曾达 24–27 GB。正式训练前仍需检查显存；调整 Batch 时按复现记录同步缩放学习率、MAX_ITER、WARMUP_ITERS 和 STEPS；已启动实验必须使用新 ID。LPI 本轮在首次正式启动前完成 Batch 校准，保留原 planned ID。

## LPI 五模型接入（2026-09-12）

- 正式配置：`lpi-001` / `lpi-002` 对应 sar-015/016 结构，`yolo-lpi-001`～`yolo-lpi-003` 对应 YOLO26n/11n/v8n；全部为 planned，不占用 SAR 编号。
- 数据注册：`lpi_train` / `lpi_val` / `lpi_test`，COCO `image_root` 为 `LPI_COCO`。五类、640、无增强、保留空标注；通用预训练、200 epoch、FP32、种子 40244023。
- 统一入口：`bash tools/train_lpi.sh <ID>`、`bash tools/evaluate_lpi.sh <ID>`；训练支持 `--resume` 与 `--check-only`。按用户最新要求默认允许未提交代码训练，并保存 Git 状态与源码快照；`--require-clean-git` 是可选严格模式。逐个启动，不自动组正式训练队列。
- 本机校准 Batch：RepViT 16、MobileNetV4 24、YOLO26n 72、YOLO11n 80、YOLOv8n 96。可用显存预检下限 20 GiB；容量与原生链路证据见日志第 6.10 节。DiffusionDet 的 LR/Iter/调度已按 200 epoch 同步换算。
- `tools/lpi_yolo.py` 保证 YOLO 训练和验证 640 输入，缓存写入 `.cache/lpi_yolo_labels`；`tools/lpi_train.py` 禁用 OOM 自动降 Batch。不要直接用通用 `yolo detect train` 绕过 LPI 适配。
- 评价工具：`tools/lpi_predict.py` 固定逐图推理并导出覆盖清单；`tools/lpi_metrics.py` 在 val 冻结 f1/low_fa 两工作点，在 test 给出 11 档 SNR×类别 AP/Pd、背景 Pfa 与单侧 95% 精确上限。JSON/CSV 指标为 0～1。
- 4,950 测试噪声窗零虚警的上限约 6.05e-4，不能认证 1e-6；纯噪声没有 SNR。详见实验日志第 5.8 节。
- 本轮 LPI 冒烟使用独立 `lpi-test-*` / `yolo-lpi-test-*`，已完成目录不得覆盖。实际占用和下个测试编号以实验日志第 6.10 节与运行目录为准。
- 新测试包含 `tests/test_lpi_metrics.py`、`tests/test_lpi_integration.py`；涉及 Torch/COCO 的完整测试使用 `Difdet` Python，系统 `python3` 无 Torch。
