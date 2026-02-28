# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone

__all__ = ["build_resnet_fpn_backbone", "build_retinanet_resnet_fpn_backbone", "FPN"]


class FPN(Backbone):
    """
    此模块实现了 :paper:`FPN`。
    它在一些输入特征图之上构建金字塔特征。
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        square_pad=0,
    ):
        """
        Args:
            bottom_up (Backbone): 代表自底向上子网络的模块。
                必须是 :class:`Backbone` 的子类。自底向上网络生成的、并在 `in_features` 中列出的多尺度特征图
                用于生成 FPN 层。
            in_features (list[str]): 来自所附着的主干网络的输入特征图的名称。
                例如，如果主干网络生成 ["res2", "res3", "res4"]，则可以使用这些的任何*连续*子列表；
                顺序必须从高分辨率到低分辨率。
            out_channels (int): 输出特征图中的通道数。
            norm (str): 要使用的归一化。
            top_block (nn.Module or None): 如果提供，将对最后一个（最小分辨率）FPN 输出的输出执行额外操作，
                结果将扩展结果列表。top_block 进一步下采样特征图。它必须具有属性
                "num_levels"（意味着此块添加的额外 FPN 层的数量）和 "in_feature"
                （表示其输入特征的字符串，例如 p5）。
            fuse_type (str): 融合自顶向下特征和横向特征的类型。
                可以是 "sum"（默认），即逐元素相加；或 "avg"，
                即取两者的逐元素平均值。
            square_pad (int): 如果 > 0，要求输入图像填充到特定的正方形尺寸。
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # 来自自底向上网络（例如 ResNet）的特征图步幅和通道
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # 将卷积层按自顶向下顺序（从低分辨率到高分辨率）排列，
        # 以使 forward 中的自顶向下计算更清晰。
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # 返回特征名称为 "p<stage>"，如 ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block 输出特征图。
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): 将特征图名称（例如 "res5"）映射到每个特征级别的特征图张量，
                按从高到低的分辨率顺序。

        Returns:
            dict[str->Tensor]:
                从特征图名称到 FPN 特征图张量的映射，按从高到低的分辨率顺序。
                返回的特征名称遵循 FPN 论文约定："p<stage>"，其中 stage 的步幅 = 2 ** stage，
                例如 ["p2", "p3", ..., "p6"]。
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # 将特征图反转为自顶向心的顺序（从低分辨率到高分辨率）
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # 不支持对 ModuleList 进行切片 https://github.com/pytorch/pytorch/issues/47336
            # 因此我们循环遍历所有模块，但跳过第一个
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    断言每个 stride 是其前一个 stride 的 2 倍，即 "log2 连续"。
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    此模块在原始 FPN 中用于从 P5 生成下采样的 P6 特征。
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    此模块在 RetinaNet 中用于从 C5 特征生成额外的层 P6 和 P7。
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): 主干模块，必须是 :class:`Backbone` 的子类。
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): 主干模块，必须是 :class:`Backbone` 的子类。
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
