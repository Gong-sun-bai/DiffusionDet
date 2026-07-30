# RTX 2080 Ti 基线环境恢复与冒烟复现记录

> 执行日期：2026-07-23
> 工作站：`troy-compute`
> 仓库：`/home/troy/workspace/DiffusionDet`
> 验证范围：SAR ResNet-18 / FPN 128 固定种子基线的环境、数据、建模、反向传播与检查点链路

## 1. 结论

当前机器已经具备运行 DiffusionDet SAR 基线的完整环境：

- Conda 环境：`/home/troy/anaconda3/envs/Difdet`
- Python：3.10.19
- PyTorch：2.10.0+cu128
- TorchVision：0.25.0+cu128
- Detectron2：仓库内 0.6
- GPU：NVIDIA GeForce RTX 2080 Ti，22,528 MiB
- 驱动：570.211.01
- 本地 Detectron2 `_C`：按 CPU-only 方式重新编译，已通过导入

`sar-007` 已完成 20 次真实训练迭代，Batch 16 未发生 OOM。该运行只证明当前机器可以启动正式基线训练，不提供 AP 或收敛性证据。正式固定种子基线仍是 `sar-005`，其输出目录未被创建。

## 2. 环境创建

仓库的 `environment.yml` 锁定 Python 3.10、PyTorch 2.10、TorchVision 0.25 和运行必需依赖，不再包含无法从 PyPI 安装的 `detectron2==0.6`，也不包含旧机器的 `prefix`。

全新创建：

```bash
cd /home/troy/workspace/DiffusionDet
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda env config vars set PYTHONNOUSERSITE=1 -n Difdet
conda activate Difdet
```

更新已有环境：

```bash
PYTHONNOUSERSITE=1 conda env update -n Difdet -f environment.yml --prune
conda env config vars set PYTHONNOUSERSITE=1 -n Difdet
conda activate Difdet
```

设置 `PYTHONNOUSERSITE=1` 是必要步骤。首次创建时发现 Pip 会看到 `~/.local/lib/python3.10/site-packages`；当前清单把 NumPy 2.2.6 和 Typing Extensions 4.15.0 作为 Conda 依赖预装，并持久化该变量。命令前缀用于保证首次 Pip 子步骤也与用户级包隔离。

检查环境：

```bash
python -m pip check
python -c "import torch, torchvision, fvcore, pycocotools, timm; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

## 3. 构建本地 Detectron2 扩展

本机系统 CUDA 编译器是 11.4，而 PyTorch wheel 使用 CUDA 12.8。为避免混用两个 CUDA 工具链，`tools/build_detectron2_extension.py` 只编译 Detectron2 的 C++ 源码：

```bash
conda activate Difdet
cd /home/troy/workspace/DiffusionDet
python tools/build_detectron2_extension.py build_ext --inplace
```

验证：

```bash
python -c "import detectron2; from detectron2 import _C; print(detectron2.__version__, _C.__file__, _C.has_cuda())"
python -m detectron2.utils.collect_env
```

本次结果为：

```text
detectron2 0.6
_C /home/troy/workspace/DiffusionDet/detectron2/_C.cpython-310-x86_64-linux-gnu.so
_C.has_cuda() False
```

SAR 基线使用的普通 ROIAlign 和 NMS 来自 TorchVision，已由 20-iter GPU 训练实际覆盖。CPU-only `_C` 仍提供 Detectron2 模块导入和快速 COCO evaluator 所需接口。若后续启用 rotated ROIAlign/NMS 或 deformable convolution 的 CUDA 路径，必须另行准备与 PyTorch CUDA 12.8 匹配的 CUDA Toolkit 并重编译，不能沿用当前 CPU-only 假设。

`collect_env` 可能输出一条来自系统 CUDA 11.4 `cuobjdump` 的 `Support for 'sm_2' has been removed`，但随后的环境表、sm_75 GPU 识别和训练均正常；这不是本次训练失败。

## 4. 数据与模型验证

数据集注册实测：

| 数据集 | 图片数 |
| --- | ---: |
| `sar_ship_train` | 31,783 |
| `sar_ship_val` | 7,946 |

`sar-005` 模型构建结果：

- ResNet 深度：18
- FPN 输出通道：128
- DiffusionDet 隐藏维度：128
- Proposal 数：500
- 参数量：34,462,942
- 设备：`cuda:0`
- FPN 输出：`p2–p6`，通道均为 128，stride 为 4/8/16/32/64

本机正式默认配置位于 `configs/machine/rtx2080ti-sar-r18.yaml`：

| 参数 | 默认值 |
| --- | ---: |
| `SOLVER.IMS_PER_BATCH` | 16 |
| `SOLVER.BASE_LR` | 0.000025 |
| `SOLVER.MAX_ITER` | 397288 |
| `SOLVER.STEPS` | 278102, 357559 |
| `SOLVER.WARMUP_ITERS` | 1000 |
| `SOLVER.CHECKPOINT_PERIOD` | 5000 |
| `DATALOADER.NUM_WORKERS` | 4 |
| `SOLVER.AMP.ENABLED` | False |

`sar-005` 和后续 `sar-006` 均继承这层机器配置。机器层只集中固化运行参数，模型结构、数据集和研究变量仍由具体实验配置声明。

## 5. `sar-007` 冒烟结果

运行命令：

```bash
conda activate Difdet
cd /home/troy/workspace/DiffusionDet
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-007-smoke.yaml
```

结果目录：

```text
runs/sar_ship/sar-007__r18-fpn128-h128-bs16-it20-smoke-seed40244023/
```

实测结果：

| 项目 | 结果 |
| --- | --- |
| 状态 | `completed` |
| 迭代 | 20 |
| Batch | 16 |
| 种子 | 40244023 |
| 平均 step 时间 | 1.2456 秒 |
| 训练主体时间 | 23 秒 |
| 端到端墙钟时间 | 32.63 秒 |
| 峰值显存 | 12,626 MB |
| 最后一条 `total_loss` | 25.35935，有限值 |
| `model_final.pth` | 412,894,970 bytes |
| 完整验证集评估 | 未运行 |

产物包括：

- `config.yaml`
- `log.txt`
- `metrics.json`
- `experiment_manifest.json`
- `result_summary.json`
- `last_checkpoint`
- `model_0000019.pth`
- `model_final.pth`

`last_checkpoint` 指向 `model_final.pth`，清单和结果摘要状态均为 `completed`。`metrics.json` 只有 writer 在 iter 19 写出的最终一行，这是 20-iter 配置的预期行为。损失数值尚未收敛，不应与历史最终指标比较。

该目录已非空，训练入口会拒绝不带 `--resume` 的重复运行。冒烟已经完成，不应为了“再验证一次”删除或覆盖该目录；需要新的烟测时应分配新的实验 ID。

## 6. 正式训练与恢复

启动固定种子正式基线：

```bash
conda activate Difdet
cd /home/troy/workspace/DiffusionDet
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-005.yaml
```

中断后恢复：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-005.yaml \
  --resume
```

正式训练前再次检查 `nvidia-smi`。`sar-007` 的 12,626 MB 是短运行实测峰值，不保证所有随机输入尺度和长时间训练都不会出现更高峰值；若出现真实 OOM，应保留失败清单并按下一节创建新实验，不得原地改变 `sar-005`。

## 7. 后续修改与微调规范

### 7.1 实验身份

- 正式实验必须分配新的未占用 ID，SAR 的 `sar-000` 至 `sar-007` 已占用。
- `smoke` 只验证运行链路，不进入 AP 排名，也不能充当消融基线。
- 一次实验只改变一个概念因素。结构兼容所需的成组参数必须在 `PURPOSE`、`HYPOTHESIS` 和 `CHANGES` 中说明。
- 不在模型源码中按实验 ID 写条件分支；新增行为应使用语义明确、默认保持旧行为的配置开关。
- 不通过命令行临时覆盖正式实验超参数；先创建配置并通过静态校验。

### 7.2 Batch 改变

当前参考值：

```text
B0 = 16
LR0 = 2.5e-5
I0 = 397288
W0 = 1000
```

如果仅因硬件容量改变全局 Batch，默认按以下规则保持样本总量和线性学习率口径：

```text
LR' = LR0 × B' / B0
MAX_ITER' = round(I0 × B0 / B')
WARMUP_ITERS' = round(W0 × B0 / B')
STEPS' = (round(0.70 × MAX_ITER'), round(0.90 × MAX_ITER'))
```

例如 Batch 8：

```yaml
SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.0000125
  MAX_ITER: 794576
  WARMUP_ITERS: 2000
  STEPS: (556203, 715118)
```

Batch 改变必须使用新实验 ID，并在 `CHANGES` 中同时记录 Batch、学习率、迭代数、warmup 和 milestones。若研究目标本身就是 Batch 或学习率，则不能把这些联动变化称为单因素硬件适配，应单独定义实验假设。

### 7.3 其他因素

- AMP 会改变数值路径，必须使用新实验 ID，先冒烟再正式训练。
- 输入尺度、裁剪、proposal 数、sample step、backbone、FPN/hidden 宽度和损失权重均属于研究因素。
- 调整 `DATALOADER.NUM_WORKERS` 前先做短运行验证；机器配置变化也要记录在复现文档中。
- 正式结果至少记录配置、Git 状态、环境版本、AP/AP50/AP75、峰值显存、训练时长和最终权重。
- 完成实验后更新 `docs/实验日志.md`；环境或上手命令变化时同步更新本文、交接指南、README 和 `AGENTS.md`。

## 8. 验证命令

```bash
python -m unittest tests.test_experiment -v
python tools/validate_experiment_configs.py
python tools/migrate_historical_runs.py --verify
git diff --check
```

本次最终结果以 `docs/实验日志.md` 和 `sar-007` 的运行清单为索引，模型权重与运行产物仍保存在被 Git 忽略的 `runs/`。
