#!/usr/bin/env python3
"""Static YAML and experiment metadata validation without model imports."""

from __future__ import annotations

import sys
import ast
from copy import deepcopy
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment_manager import (
    ExperimentError,
    metadata_from_cfg,
    validate_checkpoint_policy,
    validate_metadata,
    validate_training_parameters,
    validate_training_plot_policy,
    validate_weights,
)


EXPECTED_HISTORICAL_IDS = {
    "sar-000",
    "sar-001",
    "sar-002",
    "sar-003",
    "sar-004",
    "panda-000",
}
EXPECTED_SAR_FORMAL_IDS = {f"sar-{index:03d}" for index in range(7)}
EXPECTED_SAR_TEST_IDS = {"sar-test-001"}
STANDALONE_HISTORICAL_IDS = {f"sar-{index:03d}" for index in range(5)}

HISTORICAL_EXPECTATIONS = {
    "sar-000": {
        "MODEL.WEIGHTS": "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
        "MODEL.BACKBONE.NAME": "build_resnet_fpn_backbone",
        "MODEL.RESNETS.DEPTH": 50,
        "MODEL.RESNETS.RES2_OUT_CHANNELS": 256,
        "MODEL.FPN.OUT_CHANNELS": 256,
        "MODEL.ROI_BOX_HEAD.NUM_FC": 0,
        "MODEL.ROI_BOX_HEAD.FC_DIM": 1024,
        "MODEL.DiffusionDet.HIDDEN_DIM": 256,
        "SOLVER.IMS_PER_BATCH": 16,
        "SOLVER.MAX_ITER": 397288,
        "SOLVER.CHECKPOINT_PERIOD": 5000,
        "SOLVER.STEPS": (278102, 357559),
    },
    "sar-001": {
        "MODEL.WEIGHTS": "",
        "MODEL.BACKBONE.NAME": "build_resnet_fpn_backbone",
        "MODEL.RESNETS.DEPTH": 18,
        "MODEL.RESNETS.RES2_OUT_CHANNELS": 64,
        "MODEL.FPN.OUT_CHANNELS": 128,
        "MODEL.ROI_BOX_HEAD.NUM_FC": 0,
        "MODEL.ROI_BOX_HEAD.FC_DIM": 1024,
        "MODEL.DiffusionDet.HIDDEN_DIM": 128,
        "SOLVER.IMS_PER_BATCH": 16,
        "SOLVER.MAX_ITER": 397288,
        "SOLVER.CHECKPOINT_PERIOD": 5000,
        "SOLVER.STEPS": (278102, 357559),
    },
    "sar-002": {
        "MODEL.WEIGHTS": "",
        "MODEL.BACKBONE.NAME": "build_resnet_fpn_backbone",
        "MODEL.RESNETS.DEPTH": 18,
        "MODEL.RESNETS.RES2_OUT_CHANNELS": 64,
        "MODEL.FPN.OUT_CHANNELS": 64,
        "MODEL.ROI_BOX_HEAD.NUM_FC": 2,
        "MODEL.ROI_BOX_HEAD.FC_DIM": 256,
        "MODEL.DiffusionDet.HIDDEN_DIM": 64,
        "SOLVER.IMS_PER_BATCH": 64,
        "SOLVER.MAX_ITER": 99400,
        "SOLVER.CHECKPOINT_PERIOD": 5000,
        "SOLVER.STEPS": (278102, 357559),
    },
    "sar-003": {
        "MODEL.WEIGHTS": "",
        "MODEL.BACKBONE.NAME": "build_mobilenetv4_fpn_backbone",
        "MODEL.MOBILENETV4.SIZE": "small",
        "MODEL.FPN.OUT_CHANNELS": 64,
        "MODEL.ROI_BOX_HEAD.NUM_FC": 2,
        "MODEL.ROI_BOX_HEAD.FC_DIM": 256,
        "MODEL.DiffusionDet.HIDDEN_DIM": 64,
        "SOLVER.IMS_PER_BATCH": 64,
        "SOLVER.MAX_ITER": 99400,
        "SOLVER.CHECKPOINT_PERIOD": 5000,
        "SOLVER.STEPS": (69580, 89514),
    },
    "sar-004": {
        "MODEL.WEIGHTS": "",
        "MODEL.BACKBONE.NAME": "build_mobilenetv4_fpn_backbone",
        "MODEL.MOBILENETV4.SIZE": "medium",
        "MODEL.FPN.OUT_CHANNELS": 96,
        "MODEL.ROI_BOX_HEAD.NUM_FC": 2,
        "MODEL.ROI_BOX_HEAD.FC_DIM": 256,
        "MODEL.DiffusionDet.HIDDEN_DIM": 96,
        "SOLVER.IMS_PER_BATCH": 50,
        "SOLVER.MAX_ITER": 119280,
        "SOLVER.CHECKPOINT_PERIOD": 10000,
        "SOLVER.STEPS": (69580, 89514),
    },
}

COMMON_HISTORICAL_EXPECTATIONS = {
    "INPUT.MIN_SIZE_TRAIN": (256,),
    "INPUT.MAX_SIZE_TRAIN": 256,
    "INPUT.MIN_SIZE_TEST": 256,
    "INPUT.MAX_SIZE_TEST": 256,
    "INPUT.CROP.ENABLED": True,
    "INPUT.CROP.TYPE": "relative_range",
    "INPUT.CROP.SIZE": (0.9, 0.9),
    "SOLVER.BASE_LR": 0.000025,
    "SOLVER.WARMUP_ITERS": 1000,
    "SEED": -1,
}

RTX2080TI_BASE_REFERENCE = "../../machine/Rtx2080ti-Base-DiffusionDet.yaml"
RTX2080TI_EXPERIMENT_IDS = {"sar-005", "sar-006", "sar-test-001"}
COMMON_RTX2080TI_EXPECTATIONS = {
    "MODEL.WEIGHTS": "",
    "MODEL.RESNETS.DEPTH": 18,
    "MODEL.RESNETS.STRIDE_IN_1X1": False,
    "MODEL.RESNETS.RES2_OUT_CHANNELS": 64,
    "MODEL.FPN.OUT_CHANNELS": 128,
    "MODEL.DiffusionDet.NUM_PROPOSALS": 500,
    "MODEL.DiffusionDet.NUM_CLASSES": 1,
    "MODEL.DiffusionDet.HIDDEN_DIM": 128,
    "DATASETS.TRAIN": ("sar_ship_train",),
    "DATALOADER.NUM_WORKERS": 4,
    "SEED": 40244023,
    "INPUT.MIN_SIZE_TEST": 256,
    "SOLVER.AMP.ENABLED": False,
}
RTX2080TI_EXPERIMENT_EXPECTATIONS = {
    "sar-005": {
        "DATASETS.TEST": ("sar_ship_val",),
        "INPUT.MIN_SIZE_TRAIN": (256, 384, 512, 640, 768, 896, 1024),
        "INPUT.MAX_SIZE_TRAIN": 1024,
        "INPUT.MAX_SIZE_TEST": 1024,
        "SOLVER.IMS_PER_BATCH": 25,
        "SOLVER.BASE_LR": 0.000035,
        "SOLVER.MAX_ITER": 254264,
        "SOLVER.STEPS": (177985, 228838),
        "SOLVER.WARMUP_ITERS": 1000,
        "SOLVER.CHECKPOINT_PERIOD": 2000,
    },
    "sar-006": {
        "DATASETS.TEST": ("sar_ship_val",),
        "INPUT.MIN_SIZE_TRAIN": (256,),
        "INPUT.MAX_SIZE_TRAIN": 256,
        "INPUT.MAX_SIZE_TEST": 256,
        "SOLVER.IMS_PER_BATCH": 25,
        "SOLVER.BASE_LR": 0.000035,
        "SOLVER.MAX_ITER": 254264,
        "SOLVER.STEPS": (177985, 228838),
        "SOLVER.WARMUP_ITERS": 1000,
        "SOLVER.CHECKPOINT_PERIOD": 5000,
        "SOLVER.CHECKPOINT_RETENTION": "latest",
        "TEST.EVAL_PERIOD": 5000,
        "TEST.BEST_CHECKPOINT.ENABLED": True,
    },
    "sar-test-001": {
        "DATASETS.TEST": (),
        "INPUT.MIN_SIZE_TRAIN": (256, 384, 512, 640, 768, 896, 1024),
        "INPUT.MAX_SIZE_TRAIN": 1024,
        "INPUT.MAX_SIZE_TEST": 1024,
        "SOLVER.IMS_PER_BATCH": 16,
        "SOLVER.BASE_LR": 0.000025,
        "SOLVER.MAX_ITER": 20,
        "SOLVER.STEPS": (),
        "SOLVER.WARMUP_ITERS": 0,
        "SOLVER.CHECKPOINT_PERIOD": 20,
    },
}
RTX2080TI_ALLOWED_CHILD_SECTIONS = {
    "sar-005": {"SOLVER"},
    "sar-006": {"INPUT", "SOLVER", "TEST"},
    "sar-test-001": {"DATASETS", "SOLVER"},
}


def as_sequence(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return value


def nested_value(config, dotted_key):
    value = config
    for key in dotted_key.split("."):
        value = value[key]
    return value


def normalized_value(value):
    if isinstance(value, str) and value.startswith(("(", "[")):
        value = ast.literal_eval(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def flattened_leaves(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    leaves = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        leaves.update(flattened_leaves(child, child_prefix))
    return leaves


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key == "_BASE_":
            continue
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须是映射。")
    return value


def resolved_config(path: Path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError(f"_BASE_ 形成循环：{path}")
    current = load_yaml(path)
    base_value = current.get("_BASE_")
    if not base_value:
        return current
    bases = [base_value] if isinstance(base_value, str) else list(base_value)
    merged = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"{path} 的 _BASE_ 不存在：{base_path}")
        merged = deep_merge(merged, resolved_config(base_path, stack + (path,)))
    return deep_merge(merged, current)


def main() -> int:
    root = REPOSITORY_ROOT
    config_root = root / "configs"
    experiment_root = config_root / "experiments"
    errors = []

    yaml_files = sorted(config_root.rglob("*.yaml"))
    for path in yaml_files:
        try:
            resolved_config(path)
        except Exception as error:
            errors.append(f"{path.relative_to(root)}: {error}")

    ids = {}
    sar_formal_ids = set()
    sar_test_ids = set()
    for path in sorted(experiment_root.rglob("*.yaml")):
        relative = path.relative_to(root)
        try:
            raw = load_yaml(path)
            config = resolved_config(path)
            metadata = metadata_from_cfg(config)
            if metadata.experiment_id in ids:
                raise ExperimentError(
                    f"实验 ID 与 {ids[metadata.experiment_id]} 重复"
                )
            ids[metadata.experiment_id] = relative
            validate_metadata(metadata)
            if metadata.experiment_id not in EXPECTED_HISTORICAL_IDS:
                validate_training_parameters(
                    seed=int(config.get("SEED", -1)),
                    max_iter=int(config["SOLVER"]["MAX_ITER"]),
                    steps=as_sequence(config["SOLVER"].get("STEPS", ())),
                )
            validate_checkpoint_policy(config)
            validate_training_plot_policy(config)
            validate_weights(str(config.get("MODEL", {}).get("WEIGHTS", "")))
            if "OUTPUT_DIR" in raw:
                raise ExperimentError("实验配置不得手写 OUTPUT_DIR。")
            if metadata.dataset == "sar_ship":
                if "-test-" in metadata.experiment_id:
                    sar_test_ids.add(metadata.experiment_id)
                else:
                    sar_formal_ids.add(metadata.experiment_id)
            if metadata.experiment_id in STANDALONE_HISTORICAL_IDS:
                if "_BASE_" in raw:
                    raise ExperimentError("SAR 历史配置必须完全自包含，不得使用 _BASE_。")
                expectations = {
                    **COMMON_HISTORICAL_EXPECTATIONS,
                    **HISTORICAL_EXPECTATIONS[metadata.experiment_id],
                }
                for key, expected in expectations.items():
                    actual = normalized_value(nested_value(raw, key))
                    if actual != expected:
                        raise ExperimentError(
                            f"历史训练值不一致：{key} expected={expected!r}, "
                            f"actual={actual!r}"
                        )
            if metadata.experiment_id in RTX2080TI_EXPERIMENT_IDS:
                if raw.get("_BASE_") != RTX2080TI_BASE_REFERENCE:
                    raise ExperimentError(
                        "本机 RTX 2080 Ti 实验必须直接继承 "
                        f"{RTX2080TI_BASE_REFERENCE}。"
                    )
                allowed_sections = {
                    "_BASE_",
                    "EXPERIMENT",
                    *RTX2080TI_ALLOWED_CHILD_SECTIONS[
                        metadata.experiment_id
                    ],
                }
                unexpected_sections = set(raw) - allowed_sections
                if unexpected_sections:
                    raise ExperimentError(
                        "本机实验子配置含非差异顶层字段："
                        f"{sorted(unexpected_sections)}"
                    )
                expectations = {
                    **COMMON_RTX2080TI_EXPECTATIONS,
                    **RTX2080TI_EXPERIMENT_EXPECTATIONS[
                        metadata.experiment_id
                    ],
                }
                for key, expected in expectations.items():
                    actual = normalized_value(nested_value(config, key))
                    if actual != expected:
                        raise ExperimentError(
                            f"本机冻结训练值不一致：{key} "
                            f"expected={expected!r}, actual={actual!r}"
                        )
                base_path = (path.parent / raw["_BASE_"]).resolve()
                base_config = resolved_config(base_path)
                child_values = {
                    key: value
                    for key, value in raw.items()
                    if key not in {"_BASE_", "EXPERIMENT"}
                }
                for key, value in flattened_leaves(child_values).items():
                    try:
                        inherited = nested_value(base_config, key)
                    except KeyError:
                        continue
                    if normalized_value(value) == normalized_value(inherited):
                        raise ExperimentError(
                            f"本机实验子配置包含与父配置相同的冗余参数：{key}"
                        )
        except Exception as error:
            errors.append(f"{relative}: {error}")

    missing_historical_ids = EXPECTED_HISTORICAL_IDS - set(ids)
    if missing_historical_ids:
        errors.append(
            "历史配置集合不完整："
            f"missing={sorted(missing_historical_ids)}"
        )
    if sar_formal_ids != EXPECTED_SAR_FORMAL_IDS:
        errors.append(
            "SAR 正式实验编号不连续："
            f"expected={sorted(EXPECTED_SAR_FORMAL_IDS)}, "
            f"actual={sorted(sar_formal_ids)}"
        )
    if sar_test_ids != EXPECTED_SAR_TEST_IDS:
        errors.append(
            "SAR 测试实验编号不符合预期："
            f"expected={sorted(EXPECTED_SAR_TEST_IDS)}, "
            f"actual={sorted(sar_test_ids)}"
        )

    if errors:
        print("静态配置验证失败：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"静态配置验证通过：解析 {len(yaml_files)} 个 YAML，"
        f"检查 {len(ids)} 个实验 ID。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
