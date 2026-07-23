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
    validate_metadata,
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


def as_sequence(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
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
            validate_weights(str(config.get("MODEL", {}).get("WEIGHTS", "")))
            if "OUTPUT_DIR" in raw:
                raise ExperimentError("实验配置不得手写 OUTPUT_DIR。")
            if metadata.kind == "historical":
                historical_ids.add(metadata.experiment_id)
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
