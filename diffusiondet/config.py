# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6
    cfg.MODEL.DiffusionDet.HEAD_SHARING = "none"
    cfg.MODEL.DiffusionDet.INFERENCE_HEAD_STAGE = -1

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1
    cfg.MODEL.DiffusionDet.DDIM_ETA = 1.0
    cfg.MODEL.DiffusionDet.BOX_RENEWAL = True
    cfg.MODEL.DiffusionDet.RENEWAL_THRESHOLD = 0.5
    cfg.MODEL.DiffusionDet.ENSEMBLE_MODE = "legacy_nonfinal"

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # Generic timm features-only backbone. The selected model must expose four
    # feature levels with reductions 4/8/16/32.
    cfg.MODEL.TIMM = CN()
    cfg.MODEL.TIMM.NAME = "mobilenetv4_conv_small.e3600_r256_in1k"
    cfg.MODEL.TIMM.PRETRAINED = False
    cfg.MODEL.TIMM.OUT_INDICES = (1, 2, 3, 4)
    cfg.MODEL.TIMM.OUT_FEATURES = ("stage1", "stage2", "stage3", "stage4")

    # Optional query-consistent teacher distillation. The teacher is created
    # lazily during training and is excluded from the student state_dict.
    cfg.MODEL.DiffusionDet.DISTILLATION = CN()
    cfg.MODEL.DiffusionDet.DISTILLATION.ENABLED = False
    cfg.MODEL.DiffusionDet.DISTILLATION.TEACHER_WEIGHTS = ""
    cfg.MODEL.DiffusionDet.DISTILLATION.CONFIDENCE_THRESHOLD = 0.5
    cfg.MODEL.DiffusionDet.DISTILLATION.TEMPERATURE = 2.0
    cfg.MODEL.DiffusionDet.DISTILLATION.CLS_WEIGHT = 1.0
    cfg.MODEL.DiffusionDet.DISTILLATION.L1_WEIGHT = 1.0
    cfg.MODEL.DiffusionDet.DISTILLATION.GIOU_WEIGHT = 1.0

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    # Keep the upstream checkpoint behavior unless an experiment explicitly
    # requests a single overwritten latest checkpoint.
    cfg.SOLVER.CHECKPOINT_RETENTION = "all"

    # Optional validation-driven best checkpoint.
    cfg.TEST.BEST_CHECKPOINT = CN()
    cfg.TEST.BEST_CHECKPOINT.ENABLED = False
    cfg.TEST.BEST_CHECKPOINT.METRIC = "bbox/AP"
    cfg.TEST.BEST_CHECKPOINT.MODE = "max"

    # YOLO-style training dashboard written to OUTPUT_DIR/results.png.
    cfg.TRAINING_PLOTS = CN()
    cfg.TRAINING_PLOTS.ENABLED = True
    cfg.TRAINING_PLOTS.PERIOD = 200

    # Project experiment metadata. Every config launched through train_net.py
    # must override the blank identifiers below.
    cfg.EXPERIMENT = CN()
    cfg.EXPERIMENT.ID = ""
    cfg.EXPERIMENT.NAME = ""
    cfg.EXPERIMENT.DATASET = ""
    cfg.EXPERIMENT.DESCRIPTION = ""
    cfg.EXPERIMENT.OUTPUT_ROOT = "./runs"

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000],
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
