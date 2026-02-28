# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict
import torch.nn as nn

from detectron2.layers import ShapeSpec

__all__ = ["Backbone"]


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    网络主干的抽象基类。
    """

    def __init__(self):
        """
        任何子类的 `__init__` 方法都可以指定其自己的一组参数。
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        子类必须重写此方法，但要遵守相同的返回类型。

        Returns:
            dict[str->Tensor]: 从特征名称（例如 "res2"）到张量的映射
        """
        pass

    @property
    def size_divisibility(self) -> int:
        """
        一些主干网络要求输入的高度和宽度必须是一个特定整数的倍数。
        这对于具有横向连接的编码器/解码器类型的网络（例如 FPN）通常是正确的，
        因为特征图需要在“自底向上”和“自顶向下”的路径中匹配维度。
        如果没有特定的输入尺寸整除性要求，则设置为 0。
        """
        return 0

    @property
    def padding_constraints(self) -> Dict[str, int]:
        """
        此属性是 size_divisibility 的一种泛化。一些主干网络和训练配方需要特定的填充约束，
        例如强制被特定整数整除（例如 FPN）或填充为正方形（例如 ViTDet 中的大尺度抖动 :paper:vitdet）。
        `padding_constraints` 包含这些可选项目，例如：
        {
            "size_divisibility": int,
            "square_size": int,
            # 未来可能有更多选项
        }
        如果有 `size_divisibility`，将从这里读取，如果 `square_size` > 0，则 `square_size` 表示正方形填充尺寸。

        待办事项：使用 Dict[str, int] 类型以避免 torchscipt 问题。padding_constraints 的类型
        可以泛化为 TypedDict (Python 3.8+) 以在未来支持更多类型。
        """
        return {}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # 这是一个向后兼容的默认值
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
