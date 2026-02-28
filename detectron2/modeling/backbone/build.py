# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from .backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
主干网络注册表，用于从图像中提取特征图

注册的对象必须是接受两个参数的可调用对象：

1. 一个 :class:`detectron2.config.CfgNode`
2. 一个 :class:`detectron2.layers.ShapeSpec`，包含输入形状规范。

注册的对象必须返回 :class:`Backbone` 的实例。
"""


def build_backbone(cfg, input_shape=None):
    """
    从 `cfg.MODEL.BACKBONE.NAME` 构建主干网络。

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
