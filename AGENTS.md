# Codex 项目上下文

本文件适用于整个仓库。开始工作前先阅读 `docs/项目交接与重启指南.md`；该文档包含完整实验表、环境状态和恢复命令。

## 项目定位

本仓库是 DiffusionDet 的本地研究分支，主要研究：

- SAR 船舶单类别检测；
- PANDA 四类别检测；
- ResNet-50、ResNet-18 与 MobileNetV4 的精度/体量权衡。

上游 DiffusionDet 代码位于 `diffusiondet/`，本地还内置了 `detectron2/` 源码和 Python 3.10 编译扩展。MobileNetV4 的实现和注册位于 `diffusiondet/mobilenetv4.py`。

默认使用简体中文与用户协作，代码标识符、命令和路径保持原样。

## 当前机器和环境

- 仓库根目录：`/home/troy/workspace/DiffusionDet`
- GPU：RTX 2080 Ti，`nvidia-smi` 报告 22,528 MiB
- 当前系统 Python 缺少 PyTorch；`yolo` 环境也不是完整项目环境
- `environment.yml` 描述旧机器的 `Difdet` 环境，但尾部含 `/home/bai/...` 旧 prefix
- 运行项目前应激活/创建 `Difdet`，并同时验证 `torch.cuda.is_available()` 和 `from detectron2 import _C`

不要默认当前 shell 已处于可训练环境，也不要用系统 `/usr/bin/python3` 判断项目运行结果。

## 数据集与注册

当前实际数据目录：

- `SAR_COCO_ship/`：31,783 张训练图、7,946 张验证图、单类 `ship`
- `panda_coco_data/`：3,925 张训练图、562 张验证图、四类别

`train_net.py` 目前只启用 PANDA 注册。SAR 注册被注释且仍指向不存在的 `SAR_COCO`；SAR 工作前需改成 `SAR_COCO_ship` 并启用。建议同时注册两个数据集，不要靠来回注释切换。

相对数据路径依赖当前工作目录，运行命令时先进入仓库根目录。

## 入口和配置

- 训练/评估：`train_net.py`
- 可视化推理：`demo.py`
- 当前源配置：`configs/`
- 历史实验真实配置：优先使用对应 `output*/config.yaml`
- MobileNetV4 说明：`README_MobileNetV4.md`

配置加载顺序是 DiffusionDet 配置、MobileNetV4 配置、EMA 配置、YAML、命令行覆盖项。`--resume` 根据 `cfg.OUTPUT_DIR/last_checkpoint` 恢复。

历史 `output_1`、`output_2`、`output_3` 的配置快照仍写 `OUTPUT_DIR: ./output`；恢复时必须覆盖成各自真实目录。

## 实验事实速查

SAR 最终 AP：

- `output_1`：ResNet-50 / 110.67M 参数 / AP 64.53
- `output_2`：ResNet-18-128 / 34.49M 参数 / AP 66.94，当前最优
- `output_3`：ResNet-18-64 / 17.88M 参数 / AP 60.96
- `output_mobilenetv4_small`：7.70M 参数 / AP 54.96
- `output_mobilenetv4_medium`：20.58M 参数 / AP 57.61

`output/` 是 PANDA 四类别实验，AP 13.24，不能与 SAR 横比。

`output_3` 的 `STEPS` 大于 `MAX_ITER`，实际没有按计划降学习率。MobileNetV4-Medium 的 Max iter 被延长后，milestones 也没有保持注释所称的 70%/90%。

## 结果证据优先级

遇到文档、配置和日志冲突时：

1. 实际命令、权重加载和运行配置：看 `log.txt`
2. 最终 AP：看 `metrics.json` 最后一条带 `bbox/AP` 的记录，并用日志复核
3. 模型结构：看输出目录的冻结配置
4. 参数量：从 `model_final.pth` 的 `model` 状态字典统计
5. 当前准备启动的实验：看当前 `configs/` 和代码
6. README 和注释可能过时，只作为辅助

特别注意：`inference/instances_predictions.pth` 是逐图预测列表，不是模型权重。评估模型必须使用 `model_final.pth` 或其他模型 checkpoint。

## 安全操作规则

- 不要删除、移动或清空被 Git 忽略的 `output*`、数据集、`lwb_models/`、权重和预测文件，除非用户明确要求。
- 不要覆盖六个既有实验目录。复评使用新的 `OUTPUT_DIR <run>/recheck`。
- 工作区初始就有用户状态：`detectron2/model_zoo/configs` 为删除状态，`environment.yml` 未跟踪。不要擅自恢复或提交。
- 原 `detectron2/model_zoo/configs` 是指向 `/home/bai/...` 的失效绝对符号链接，修复前先确认真正需要的目标。
- 历史训练曾使用 24–27 GB 显存；当前 GPU 报告 22.5 GB。开训前先降低 Batch 做显存探测。
- 修改 Batch 后同步审查 `MAX_ITER`、`STEPS`、学习率和 epoch 口径。
- 新实验必须使用新的、有含义的输出目录，并固定 `SEED`。

## 已知硬编码和待清理项

- `ship_coco.py` 和 `param_compare.py` 含 `/home/bai/demo/DiffusionDet` 绝对路径。
- `README.md` 的 SAR 评估示例错误地使用预测文件作为权重。
- `README_MobileNetV4.md` 使用旧目录名 `SAR_COCO`，部分参数量是 backbone 估计，不是完整检测模型参数。
- `diffusiondet/util/low_rank.py` 当前未被引用，不要假定低秩层已经生效。
- PANDA 包含空标注图片，默认训练加载器会过滤；计算 epoch 时要说明口径。

## 推荐验证层级

根据改动风险选择验证：

1. 纯文档或路径说明：核对文件、Git 状态和引用路径。
2. 配置/注册改动：在 `Difdet` 环境中完成模块导入、配置合并和数据集 catalog 查询。
3. 模型代码改动：构建对应模型，检查特征名/shape 和 checkpoint 加载。
4. 训练逻辑改动：先做短迭代 smoke test，使用全新临时输出目录。
5. 结果结论：复评到新的 `recheck` 目录，对照历史 AP，绝不覆盖历史证据。

一个安全的 SAR 复评模板（需先修好 SAR 注册）：

```bash
python train_net.py --num-gpus 1 \
  --config-file output_2/config.yaml \
  --eval-only \
  MODEL.WEIGHTS output_2/model_final.pth \
  OUTPUT_DIR output_2/recheck
```

完成新实验或修正关键项目事实后，同步更新 `docs/项目交接与重启指南.md`，记录配置、输出目录、指标、验证方式和仍存在的限制。
