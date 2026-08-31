"""Generic timm features-only backbone adapter for Detectron2 FPN."""

from __future__ import annotations

from collections import OrderedDict

import timm

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool


class TimmBackbone(Backbone):
    """Expose a four-level timm feature extractor as a Detectron2 backbone."""

    def __init__(self, cfg):
        super().__init__()
        name = str(cfg.MODEL.TIMM.NAME)
        out_indices = tuple(int(value) for value in cfg.MODEL.TIMM.OUT_INDICES)
        out_features = tuple(str(value) for value in cfg.MODEL.TIMM.OUT_FEATURES)
        if len(out_indices) != 4 or len(out_features) != 4:
            raise ValueError("MODEL.TIMM must select exactly four output levels")

        self.model_name = name
        self.pretrained = bool(cfg.MODEL.TIMM.PRETRAINED)
        self.out_features = out_features
        self.bottom_up = timm.create_model(
            name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels = tuple(int(value) for value in self.bottom_up.feature_info.channels())
        reductions = tuple(int(value) for value in self.bottom_up.feature_info.reduction())
        if reductions != (4, 8, 16, 32):
            raise ValueError(
                f"timm backbone {name!r} reductions must be (4, 8, 16, 32), "
                f"got {reductions}"
            )
        self._out_feature_channels = dict(zip(out_features, channels))
        self._out_feature_strides = dict(zip(out_features, reductions))
        self._size_divisibility = 32

    def forward(self, x):
        outputs = self.bottom_up(x)
        if len(outputs) != len(self.out_features):
            raise RuntimeError(
                f"timm backbone returned {len(outputs)} levels, "
                f"expected {len(self.out_features)}"
            )
        return OrderedDict(zip(self.out_features, outputs))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self.out_features
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility


@BACKBONE_REGISTRY.register()
def build_timm_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = TimmBackbone(cfg)
    return FPN(
        bottom_up=bottom_up,
        in_features=list(cfg.MODEL.FPN.IN_FEATURES),
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
