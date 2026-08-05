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
    historical_ids = set()
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
            validate_metadata(
                metadata,
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
            if metadata.kind == "historical":
                historical_ids.add(metadata.experiment_id)
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
            if metadata.kind == "ablation":
                if metadata.baseline not in ids and not any(
                    candidate.name.startswith(metadata.baseline)
                    for candidate in experiment_root.rglob("*.yaml")
                ):
                    raise ExperimentError(
                        f"消融基线不存在：{metadata.baseline}"
                    )
        except Exception as error:
            errors.append(f"{relative}: {error}")

    if historical_ids != EXPECTED_HISTORICAL_IDS:
        errors.append(
            "历史配置集合不完整："
            f"expected={sorted(EXPECTED_HISTORICAL_IDS)}, "
            f"actual={sorted(historical_ids)}"
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
