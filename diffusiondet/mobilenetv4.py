# ========================================
# MobileNetV4 Backbone for DiffusionDet
# 参考：https://arxiv.org/abs/2404.10518
# ========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, List, Optional

import fvcore.nn.weight_init as weight_init
from detectron2.layers import ShapeSpec, get_norm
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool


__all__ = [
    "MobileNetV4",
    "build_mobilenetv4_backbone",
    "build_mobilenetv4_fpn_backbone",
]


def make_divisible(value, divisor=8, min_value=None, min_ratio=0.9):
    """确保所有层的通道数都可被 divisor 整除"""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class ConvBNAct(nn.Module):
    """卷积 + BatchNorm + 激活函数"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        norm: str = "BN",
        act: nn.Module = nn.ReLU6,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = get_norm(norm, out_channels)
        self.act = act(inplace=True) if act is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation 模块"""

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        act_layer: nn.Module = nn.ReLU,
        gate_layer: nn.Module = nn.Sigmoid,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class UniversalInvertedBottleneck(nn.Module):
    """
    MobileNetV4 的 Universal Inverted Bottleneck (UIB) 模块
    支持多种配置：ExtraDW, ConvNext, IB（Inverted Bottleneck）, FFN
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float,
        stride: int = 1,
        kernel_size: int = 3,
        middle_dw: bool = True,
        start_dw: bool = False,
        start_dw_kernel_size: int = 0,
        use_se: bool = False,
        se_ratio: float = 0.25,
        norm: str = "BN",
        act: nn.Module = nn.ReLU6,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        mid_channels = make_divisible(in_channels * expand_ratio)
        
        layers = []
        
        # 可选的起始 Depthwise Conv（用于 ExtraDW 配置）
        if start_dw and start_dw_kernel_size > 0:
            layers.append(ConvBNAct(
                in_channels, in_channels,
                kernel_size=start_dw_kernel_size,
                stride=stride,
                groups=in_channels,
                norm=norm,
                act=act,
            ))
            stride = 1  # stride 已经在 start_dw 中处理
        
        # Expansion 1x1 Conv
        if expand_ratio != 1:
            layers.append(ConvBNAct(
                in_channels, mid_channels,
                kernel_size=1,
                norm=norm,
                act=act,
            ))
        else:
            mid_channels = in_channels
        
        # 中间的 Depthwise Conv（用于标准 IB 配置）
        if middle_dw:
            layers.append(ConvBNAct(
                mid_channels, mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=mid_channels,
                norm=norm,
                act=act,
            ))
        
        # Squeeze-and-Excitation
        if use_se:
            squeeze_channels = make_divisible(mid_channels * se_ratio)
            layers.append(SqueezeExcite(mid_channels, squeeze_channels))
        
        # Projection 1x1 Conv (无激活)
        layers.append(ConvBNAct(
            mid_channels, out_channels,
            kernel_size=1,
            norm=norm,
            act=None,
        ))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetV4Block(nn.Module):
    """
    MobileNetV4 单个 Block
    用于封装 ConvBN 或 UIB 模块
    """

    def __init__(self, block_type: str, **kwargs):
        super().__init__()
        self.block_type = block_type
        
        if block_type == "convbn":
            self.block = ConvBNAct(**kwargs)
        elif block_type == "uib":
            self.block = UniversalInvertedBottleneck(**kwargs)
        elif block_type == "fused_ib":
            # Fused Inverted Bottleneck（使用 3x3 Conv 替代 DW + PW）
            self.block = self._make_fused_ib(**kwargs)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def _make_fused_ib(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float,
        stride: int = 1,
        kernel_size: int = 3,
        norm: str = "BN",
        act: nn.Module = nn.ReLU6,
        **kwargs,
    ):
        """Fused Inverted Bottleneck"""
        mid_channels = make_divisible(in_channels * expand_ratio)
        layers = []
        
        # Fused expand conv
        layers.append(ConvBNAct(
            in_channels, mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm,
            act=act,
        ))
        
        # Projection
        layers.append(ConvBNAct(
            mid_channels, out_channels,
            kernel_size=1,
            norm=norm,
            act=None,
        ))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# =====================================
# MobileNetV4 配置
# =====================================

# MobileNetV4-Small 配置
MOBILENETV4_SMALL_SPEC = {
    "stem": {"out_channels": 32, "stride": 2, "kernel_size": 3},
    "stages": [
        # Stage 1 (stride=2, output_stride=4)
        [
            {"type": "fused_ib", "in": 32, "out": 32, "expand": 4.0, "stride": 2, "kernel": 3},
            {"type": "fused_ib", "in": 32, "out": 32, "expand": 2.0, "stride": 1, "kernel": 3},
        ],
        # Stage 2 (stride=2, output_stride=8)
        [
            {"type": "fused_ib", "in": 32, "out": 64, "expand": 4.0, "stride": 2, "kernel": 3},
            {"type": "fused_ib", "in": 64, "out": 64, "expand": 2.0, "stride": 1, "kernel": 3},
        ],
        # Stage 3 (stride=2, output_stride=16)
        [
            {"type": "uib", "in": 64, "out": 96, "expand": 4.0, "stride": 2, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 96, "out": 96, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 96, "out": 96, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 96, "out": 96, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
        ],
        # Stage 4 (stride=2, output_stride=32)
        [
            {"type": "uib", "in": 96, "out": 128, "expand": 4.0, "stride": 2, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 128, "out": 128, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 128, "out": 128, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 128, "out": 128, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
        ],
    ],
    "out_channels": [32, 64, 96, 128],  # 各阶段输出通道数
}

# MobileNetV4-Medium 配置
MOBILENETV4_MEDIUM_SPEC = {
    "stem": {"out_channels": 32, "stride": 2, "kernel_size": 3},
    "stages": [
        # Stage 1 (stride=2, output_stride=4)
        [
            {"type": "fused_ib", "in": 32, "out": 48, "expand": 4.0, "stride": 2, "kernel": 3},
            {"type": "fused_ib", "in": 48, "out": 48, "expand": 2.0, "stride": 1, "kernel": 3},
        ],
        # Stage 2 (stride=2, output_stride=8)
        [
            {"type": "fused_ib", "in": 48, "out": 80, "expand": 4.0, "stride": 2, "kernel": 3},
            {"type": "fused_ib", "in": 80, "out": 80, "expand": 2.0, "stride": 1, "kernel": 3},
        ],
        # Stage 3 (stride=2, output_stride=16)
        [
            {"type": "uib", "in": 80, "out": 160, "expand": 4.0, "stride": 2, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 160, "out": 160, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 160, "out": 160, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 160, "out": 160, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 160, "out": 160, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 160, "out": 160, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 160, "out": 160, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": False},
            {"type": "uib", "in": 160, "out": 160, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
        ],
        # Stage 4 (stride=2, output_stride=32)
        [
            {"type": "uib", "in": 160, "out": 256, "expand": 6.0, "stride": 2, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 256, "out": 256, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 256, "out": 256, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 256, "out": 256, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 256, "out": 256, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": False},
            {"type": "uib", "in": 256, "out": 256, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": False},
            {"type": "uib", "in": 256, "out": 256, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 256, "out": 256, "expand": 2.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 256, "out": 256, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 256, "out": 256, "expand": 2.0, "stride": 1, "kernel": 5, "start_dw": False},
        ],
    ],
    "out_channels": [48, 80, 160, 256],
}

# MobileNetV4-Large 配置
MOBILENETV4_LARGE_SPEC = {
    "stem": {"out_channels": 24, "stride": 2, "kernel_size": 3},
    "stages": [
        # Stage 1 (stride=2, output_stride=4)
        [
            {"type": "fused_ib", "in": 24, "out": 48, "expand": 4.0, "stride": 2, "kernel": 3},
            {"type": "fused_ib", "in": 48, "out": 48, "expand": 2.0, "stride": 1, "kernel": 3},
        ],
        # Stage 2 (stride=2, output_stride=8)
        [
            {"type": "fused_ib", "in": 48, "out": 96, "expand": 4.0, "stride": 2, "kernel": 3},
            {"type": "fused_ib", "in": 96, "out": 96, "expand": 2.0, "stride": 1, "kernel": 3},
        ],
        # Stage 3 (stride=2, output_stride=16)
        [
            {"type": "uib", "in": 96, "out": 192, "expand": 4.0, "stride": 2, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": False},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": True, "start_dw_ks": 3},
            {"type": "uib", "in": 192, "out": 192, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
        ],
        # Stage 4 (stride=2, output_stride=32)
        [
            {"type": "uib", "in": 192, "out": 512, "expand": 6.0, "stride": 2, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": False},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 3, "start_dw": False},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": True, "start_dw_ks": 5},
            {"type": "uib", "in": 512, "out": 512, "expand": 4.0, "stride": 1, "kernel": 5, "start_dw": False},
        ],
    ],
    "out_channels": [48, 96, 192, 512],
}


# 配置映射
MOBILENETV4_SPECS = {
    "small": MOBILENETV4_SMALL_SPEC,
    "medium": MOBILENETV4_MEDIUM_SPEC,
    "large": MOBILENETV4_LARGE_SPEC,
    "S": MOBILENETV4_SMALL_SPEC,
    "M": MOBILENETV4_MEDIUM_SPEC,
    "L": MOBILENETV4_LARGE_SPEC,
}


class MobileNetV4(Backbone):
    """
    MobileNetV4 主干网络
    
    参考论文: "MobileNetV4: Universal Models for the Mobile Ecosystem"
    https://arxiv.org/abs/2404.10518
    """

    def __init__(
        self,
        size: str = "small",
        in_channels: int = 3,
        out_features: List[str] = None,
        norm: str = "BN",
        width_multiplier: float = 1.0,
        freeze_at: int = 0,
    ):
        """
        Args:
            size: 模型大小 ("small", "medium", "large" 或 "S", "M", "L")
            in_channels: 输入通道数
            out_features: 输出特征层名称列表，如 ["stage1", "stage2", "stage3", "stage4"]
            norm: 归一化类型
            width_multiplier: 宽度乘数，用于缩放通道数
            freeze_at: 冻结的阶段数量（0=不冻结, 1=冻结stem, 2=冻结stem+stage1, ...）
        """
        super().__init__()
        
        spec = MOBILENETV4_SPECS[size]
        self.norm = norm
        
        # 默认输出所有阶段的特征
        if out_features is None:
            out_features = ["stage1", "stage2", "stage3", "stage4"]
        self._out_features = out_features
        
        # 构建 stem
        stem_channels = make_divisible(spec["stem"]["out_channels"] * width_multiplier)
        self.stem = ConvBNAct(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=spec["stem"]["kernel_size"],
            stride=spec["stem"]["stride"],
            norm=norm,
            act=nn.ReLU6,
        )
        
        # 构建各个 stage
        self.stages = nn.ModuleList()
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        
        current_stride = 2  # stem 的 stride
        in_ch = stem_channels
        
        for stage_idx, stage_spec in enumerate(spec["stages"]):
            stage_name = f"stage{stage_idx + 1}"
            blocks = []
            
            for block_spec in stage_spec:
                out_ch = make_divisible(block_spec["out"] * width_multiplier)
                
                if block_spec["type"] == "fused_ib":
                    blocks.append(MobileNetV4Block(
                        block_type="fused_ib",
                        in_channels=in_ch,
                        out_channels=out_ch,
                        expand_ratio=block_spec["expand"],
                        stride=block_spec["stride"],
                        kernel_size=block_spec["kernel"],
                        norm=norm,
                    ))
                elif block_spec["type"] == "uib":
                    start_dw = block_spec.get("start_dw", False)
                    start_dw_ks = block_spec.get("start_dw_ks", 0)
                    blocks.append(MobileNetV4Block(
                        block_type="uib",
                        in_channels=in_ch,
                        out_channels=out_ch,
                        expand_ratio=block_spec["expand"],
                        stride=block_spec["stride"],
                        kernel_size=block_spec["kernel"],
                        start_dw=start_dw,
                        start_dw_kernel_size=start_dw_ks,
                        norm=norm,
                    ))
                
                if block_spec["stride"] > 1:
                    current_stride *= block_spec["stride"]
                in_ch = out_ch
            
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            
            # 记录输出特征信息
            self._out_feature_channels[stage_name] = out_ch
            self._out_feature_strides[stage_name] = current_stride
        
        # 权重初始化
        self._init_weights()
        
        # 冻结指定阶段
        self._freeze_stages(freeze_at)

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_stages(self, freeze_at: int):
        """冻结指定阶段的参数"""
        if freeze_at >= 1:
            for param in self.stem.parameters():
                param.requires_grad = False
        
        for stage_idx in range(min(freeze_at - 1, len(self.stages))):
            for param in self.stages[stage_idx].parameters():
                param.requires_grad = False

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像张量 (N, C, H, W)
            
        Returns:
            dict[str, Tensor]: 特征图字典，键为 "stage1", "stage2", "stage3", "stage4"
        """
        outputs = {}
        x = self.stem(x)
        
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            stage_name = f"stage{stage_idx + 1}"
            if stage_name in self._out_features:
                outputs[stage_name] = x
        
        return outputs

    def output_shape(self) -> Dict[str, ShapeSpec]:
        """返回输出特征的形状规格"""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self) -> int:
        """输入尺寸需要被整除的数"""
        return 32


# =====================================
# 注册到 detectron2 骨干网络注册表
# =====================================

@BACKBONE_REGISTRY.register()
def build_mobilenetv4_backbone(cfg, input_shape: ShapeSpec):
    """
    从配置构建 MobileNetV4 骨干网络
    
    Args:
        cfg: detectron2 配置节点
        input_shape: 输入形状规格
        
    Returns:
        MobileNetV4 实例
    """
    size = cfg.MODEL.MOBILENETV4.SIZE
    out_features = cfg.MODEL.MOBILENETV4.OUT_FEATURES
    norm = cfg.MODEL.MOBILENETV4.NORM
    width_multiplier = cfg.MODEL.MOBILENETV4.WIDTH_MULTIPLIER
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    
    return MobileNetV4(
        size=size,
        in_channels=input_shape.channels,
        out_features=out_features,
        norm=norm,
        width_multiplier=width_multiplier,
        freeze_at=freeze_at,
    )


@BACKBONE_REGISTRY.register()
def build_mobilenetv4_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    构建带 FPN 的 MobileNetV4 骨干网络
    
    Args:
        cfg: detectron2 配置节点
        input_shape: 输入形状规格
        
    Returns:
        带 FPN 的 MobileNetV4 实例
    """
    bottom_up = build_mobilenetv4_backbone(cfg, input_shape)
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


def add_mobilenetv4_config(cfg):
    """
    向配置添加 MobileNetV4 相关的配置项
    
    Args:
        cfg: detectron2 配置节点
    """
    from detectron2.config import CfgNode as CN
    
    cfg.MODEL.MOBILENETV4 = CN()
    cfg.MODEL.MOBILENETV4.SIZE = "small"  # "small", "medium", "large"
    cfg.MODEL.MOBILENETV4.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
    cfg.MODEL.MOBILENETV4.NORM = "BN"
    cfg.MODEL.MOBILENETV4.WIDTH_MULTIPLIER = 1.0
