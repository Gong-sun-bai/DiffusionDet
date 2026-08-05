## DiffusionDet: Diffusion Model for Object Detection

**DiffusionDet is the first work of diffusion model for object detection.**

![](teaser.png)


> [**DiffusionDet: Diffusion Model for Object Detection**](https://arxiv.org/abs/2211.09788)               
> [Shoufa Chen](https://www.shoufachen.com/), [Peize Sun](https://peizesun.github.io/), [Yibing Song](https://ybsong00.github.io/), [Ping Luo](http://luoping.me/)                 
> *[arXiv 2211.09788](https://arxiv.org/abs/2211.09788)* 

## Updates
- (11/2022) Code is released.

## Models
Method | Box AP (1 step) | Box AP (4 step) | Download
--- |:---:|:---:|:---:
[COCO-Res50](configs/diffdet.coco.res50.yaml) | 45.5 | 46.1 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res50.pth)
[COCO-Res101](configs/diffdet.coco.res101.yaml) | 46.6 | 46.9 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res101.pth)
[COCO-SwinBase](configs/diffdet.coco.swinbase.yaml) | 52.3 | 52.7 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_swinbase.pth)
[LVIS-Res50](configs/diffdet.lvis.res50.yaml) | 30.4 | 31.8 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_res50.pth)
[LVIS-Res101](configs/diffdet.lvis.res101.yaml) | 31.9 | 32.9 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_res101.pth)
[LVIS-SwinBase](configs/diffdet.lvis.swinbase.yaml) | 40.6 | 41.9 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_swinbase.pth)


## Getting Started

The installation instruction and usage are in [Getting Started with DiffusionDet](GETTING_STARTED.md).

## 本地实验工作流

本仓库的 SAR/PANDA 研究分支使用严格的实验元数据和自动输出目录。开始工作前请先阅读：

- [RTX 2080 Ti 基线复现记录](docs/RTX2080Ti基线复现记录.md)
- [项目交接与重启指南](docs/项目交接与重启指南.md)
- [实验日志](docs/实验日志.md)
- [Codex 项目上下文](AGENTS.md)

当前工作站已创建 `Difdet` 环境，完成 Batch 16、20 iter 的 `sar-test-001`
冒烟训练、`sar-005` 多尺度训练和 `sar-006` 固定 256 训练。首次恢复环境时执行：

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda env config vars set PYTHONNOUSERSITE=1 -n Difdet
conda activate Difdet
python tools/build_detectron2_extension.py build_ext --inplace
```

本地实验配置统一位于：

```text
configs/experiments/
├── panda/panda-000.yaml
└── sar_ship/
    ├── sar-000.yaml ... sar-004.yaml  # 独立、历史只读配置
    ├── sar-005.yaml                   # 固定种子多尺度实验
    ├── sar-006-fixed256.yaml          # 已完成的固定 256 可重复基线
    └── sar-test-001-smoke.yaml        # 已完成的本机训练冒烟
```

RTX 2080 Ti 的 Batch、学习率、训练长度、milestones、worker 和 AMP
默认值集中在 `configs/machine/rtx2080ti-sar-r18.yaml`。正式实验应继承机器
配置并分配新 ID；Batch 改变时的联动缩放规范见复现记录。

复评固定 256 基线的最佳权重：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-006-fixed256.yaml \
  --eval-only \
  MODEL.WEIGHTS runs/sar_ship/sar-006__r18-fpn128-h128-bs25-it254264-fixed256-seed40244023/model_best.pth
```

入口会根据 `EXPERIMENT.*` 自动写入
`runs/<dataset>/<ID>__<NAME>/`。新训练禁止复用非空目录；恢复必须传
`--resume` 且存在有效的 `last_checkpoint`。

训练期间默认每 200 iter 原子更新结果目录中的 `results.png`，并在周期
验证和训练结束后强制刷新。图片统一展示总损失、分类/L1/GIoU 主头损失、
学习率、迭代耗时以及已有的 bbox AP。可通过
`TRAINING_PLOTS.ENABLED` 和 `TRAINING_PLOTS.PERIOD` 关闭或调整频率。
为旧结果目录补图：

```bash
python tools/generate_training_plots.py \
  runs/sar_ship/sar-001__r18-fpn128-h128-bs16-it397288-scratch
```

补图只读取 `metrics.json` 和可选的 `config.yaml`，不会改写原始日志、
权重或实验状态。历史实验通常只在训练结束时评估一次，因此 AP 面板会显示
最终单点，而不是伪造连续验证曲线。

评估必须在命令行显式指定 `model_final.pth`、`model_latest.pth` 或
`model_best.pth`，结果自动进入原实验的时间戳子目录：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-001.yaml \
  --eval-only \
  MODEL.WEIGHTS runs/sar_ship/sar-001__r18-fpn128-h128-bs16-it397288-scratch/model_final.pth
```

`inference/instances_predictions.pth` 是预测缓存，不是模型权重。

`sar-006` 在保持 `sar-005` 的种子、Batch、学习率和步数时恢复固定 256
输入，最佳 `AP=67.7707`，相对多尺度 `sar-005` 提高 8.93 AP，现为可重复
SAR 基线。最佳精度使用 `model_best.pth`，可靠续训使用 `model_latest.pth`。
正式实验下一编号为 `sar-007`，测试实验下一编号为 `sar-test-002`。历史结果
只保存在本地 `runs/`，该目录被 Git 忽略。


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.


## Citing DiffusionDet

If you use DiffusionDet in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX
@article{chen2022diffusiondet,
      title={DiffusionDet: Diffusion Model for Object Detection},
      author={Chen, Shoufa and Sun, Peize and Song, Yibing and Luo, Ping},
      journal={arXiv preprint arXiv:2211.09788},
      year={2022}
}
```
