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

- [项目交接与重启指南](docs/项目交接与重启指南.md)
- [实验日志](docs/实验日志.md)
- [Codex 项目上下文](AGENTS.md)

本地实验配置统一位于：

```text
configs/experiments/
├── panda/panda-000.yaml
└── sar_ship/
    ├── sar-000.yaml ... sar-004.yaml  # 历史只读配置
    ├── sar-005.yaml                   # 固定种子正式基线
    └── sar-006-proposals300.yaml      # 单因素消融示例
```

启动固定种子基线：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-005.yaml
```

入口会根据 `EXPERIMENT.*` 自动写入
`runs/<dataset>/<ID>__<NAME>/`。新训练禁止复用非空目录；恢复必须传
`--resume` 且存在有效的 `last_checkpoint`。

评估必须在命令行显式指定 `model_final.pth`，结果自动进入原实验的时间戳子目录：

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/experiments/sar_ship/sar-001.yaml \
  --eval-only \
  MODEL.WEIGHTS runs/sar_ship/sar-001__r18-fpn128-h128-bs16-it397288-scratch/model_final.pth
```

`inference/instances_predictions.pth` 是预测缓存，不是模型权重。

当前历史 SAR 最高结果为 `sar-001` 的 AP 66.94，但它使用 `SEED=-1`；
后续消融必须先完成 `sar-005` 的固定种子复跑。历史结果只保存在本地
`runs/`，该目录被 Git 忽略。


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
