#!/usr/bin/env python3
"""Safely migrate the six historical output directories into runs/.

Dry-run is the default. Use --execute to move and add manifests, then --verify
to re-check the migrated layout. Raw config/log/metrics/checkpoint files are
never rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST = "experiment_manifest.json"
REQUIRED_FILES = ("config.yaml", "log.txt", "metrics.json", "last_checkpoint")
# 训练可视化是迁移后由日志派生的便捷产物，不属于最初迁移的原始证据快照。
DERIVED_FILES = {"results.png"}
EXPERIMENT_DESCRIPTIONS = {
    "sar-000": (
        "SAR ResNet-50、FPN/HIDDEN=256、ImageNet 预训练的历史精度基线；"
        "仅用于复评，不得重新训练。"
    ),
    "sar-001": (
        "相对 sar-000 改为 ResNet-18、FPN/HIDDEN=128 并从头训练，用于验证"
        "轻量化能否保持 SAR 精度；SEED=-1，仅作方向性参考。"
    ),
    "sar-002": (
        "相对 sar-001 将 FPN/HIDDEN 降至 64、Batch 调至 64、MAX_ITER 缩至 "
        "99400；SOLVER.STEPS 超过 MAX_ITER，结果状态为 invalid。"
    ),
    "sar-003": (
        "相对 sar-000 改用 MobileNetV4-Small、FPN/HIDDEN=64 并从头训练，"
        "用于轻量化对比；没有端侧速度证据。"
    ),
    "sar-004": (
        "相对 sar-003 改用 MobileNetV4-Medium、FPN/HIDDEN=96、Batch 50 并"
        "延长至 119280 iter；milestones 未随训练计划重算。"
    ),
    "panda-000": (
        "PANDA 四类别 ResNet-50/FPN256 ImageNet 预训练历史基线；不可与 "
        "SAR 单类别结果横向比较，仅用于复评。"
    ),
}

RUNS = (
    {
        "old": "output_1",
        "new": "runs/sar_ship/sar-000__r50-fpn256-h256-bs16-it397288-pretrain",
        "id": "sar-000",
        "name": "r50-fpn256-h256-bs16-it397288-pretrain",
        "dataset": "sar_ship",
        "config": "configs/experiments/sar_ship/sar-000.yaml",
        "status": "completed",
        "parameters_million": 110.67,
        "peak_memory_mb": 18825,
        "training_time": "1 day, 23:11:15",
        "parameters": {
            "backbone": "ResNet-50",
            "fpn_channels": 256,
            "hidden_dim": 256,
            "batch_size": 16,
            "max_iter": 397288,
            "steps": [278102, 357559],
            "seed": -1,
            "initial_weights": "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
        },
        "issues": [
            {
                "code": "frozen_config_overwritten",
                "detail": "失败评估覆盖过 config.yaml 的 MODEL.WEIGHTS；原训练权重来源以 log.txt 为准。",
            }
        ],
    },
    {
        "old": "output_2",
        "new": "runs/sar_ship/sar-001__r18-fpn128-h128-bs16-it397288-scratch",
        "id": "sar-001",
        "name": "r18-fpn128-h128-bs16-it397288-scratch",
        "dataset": "sar_ship",
        "config": "configs/experiments/sar_ship/sar-001.yaml",
        "status": "completed",
        "parameters_million": 34.49,
        "peak_memory_mb": 10158,
        "training_time": "1 day, 08:37:54",
        "parameters": {
            "backbone": "ResNet-18",
            "fpn_channels": 128,
            "hidden_dim": 128,
            "batch_size": 16,
            "max_iter": 397288,
            "steps": [278102, 357559],
            "seed": -1,
            "initial_weights": "",
        },
        "issues": [
            {
                "code": "non_reproducible_seed",
                "detail": "当前最高 AP 参考结果，但 SEED=-1，不作为正式可重复基线。",
            }
        ],
    },
    {
        "old": "output_3",
        "new": "runs/sar_ship/sar-002__r18-fpn64-h64-bs64-it99400-scratch",
        "id": "sar-002",
        "name": "r18-fpn64-h64-bs64-it99400-scratch",
        "dataset": "sar_ship",
        "config": "configs/experiments/sar_ship/sar-002.yaml",
        "status": "invalid",
        "parameters_million": 17.88,
        "peak_memory_mb": 24261,
        "training_time": None,
        "parameters": {
            "backbone": "ResNet-18",
            "fpn_channels": 64,
            "hidden_dim": 64,
            "batch_size": 64,
            "max_iter": 99400,
            "steps": [278102, 357559],
            "seed": -1,
            "initial_weights": "",
        },
        "issues": [
            {
                "code": "invalid_schedule",
                "detail": "SOLVER.STEPS 全部大于 MAX_ITER，训练未执行计划中的学习率衰减。",
            },
            {
                "code": "resumed_training",
                "detail": "训练从 iteration 20000 恢复过，日志片段时长不可直接相加为墙钟时长。",
            },
        ],
    },
    {
        "old": "output_mobilenetv4_small",
        "new": "runs/sar_ship/sar-003__mnv4-small-fpn64-h64-bs64-it99400-scratch",
        "id": "sar-003",
        "name": "mnv4-small-fpn64-h64-bs64-it99400-scratch",
        "dataset": "sar_ship",
        "config": "configs/experiments/sar_ship/sar-003.yaml",
        "status": "completed",
        "parameters_million": 7.70,
        "peak_memory_mb": 25003,
        "training_time": "1 day, 00:00:15",
        "parameters": {
            "backbone": "MobileNetV4-Small",
            "fpn_channels": 64,
            "hidden_dim": 64,
            "batch_size": 64,
            "max_iter": 99400,
            "steps": [69580, 89514],
            "seed": -1,
            "initial_weights": "",
        },
        "issues": [
            {
                "code": "non_reproducible_seed",
                "detail": "SEED=-1 且仅有一次运行。",
            }
        ],
    },
    {
        "old": "output_mobilenetv4_medium",
        "new": "runs/sar_ship/sar-004__mnv4-medium-fpn96-h96-bs50-it119280-scratch",
        "id": "sar-004",
        "name": "mnv4-medium-fpn96-h96-bs50-it119280-scratch",
        "dataset": "sar_ship",
        "config": "configs/experiments/sar_ship/sar-004.yaml",
        "status": "completed",
        "parameters_million": 20.58,
        "peak_memory_mb": 27403,
        "training_time": None,
        "parameters": {
            "backbone": "MobileNetV4-Medium",
            "fpn_channels": 96,
            "hidden_dim": 96,
            "batch_size": 50,
            "max_iter": 119280,
            "steps": [69580, 89514],
            "seed": -1,
            "initial_weights": "",
        },
        "issues": [
            {
                "code": "milestones_not_rescaled",
                "detail": "MAX_ITER 延长后 milestones 未同步重算，只位于约 58%/75%。",
            },
            {
                "code": "resumed_training",
                "detail": "训练多次从 iteration 50000 恢复，无法从单条日志得出完整墙钟时长。",
            },
        ],
    },
    {
        "old": "output",
        "new": "runs/panda/panda-000__r50-fpn256-h256-bs16-it49063-pretrain",
        "id": "panda-000",
        "name": "r50-fpn256-h256-bs16-it49063-pretrain",
        "dataset": "panda",
        "config": "configs/experiments/panda/panda-000.yaml",
        "status": "completed",
        "parameters_million": 110.67,
        "peak_memory_mb": 27049,
        "training_time": "10:28:54",
        "parameters": {
            "backbone": "ResNet-50",
            "fpn_channels": 256,
            "hidden_dim": 256,
            "batch_size": 16,
            "max_iter": 49063,
            "steps": [34344, 44157],
            "seed": -1,
            "initial_weights": "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
        },
        "issues": [
            {
                "code": "cross_dataset_not_comparable",
                "detail": "PANDA 是四类别任务，指标不得与 SAR 单类别实验横向比较。",
            }
        ],
    },
)


def final_bbox_metrics(metrics_file: Path) -> dict[str, Any]:
    final = None
    with metrics_file.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "bbox/AP" in record:
                final = {
                    key.removeprefix("bbox/"): value
                    for key, value in record.items()
                    if key.startswith("bbox/")
                }
                final["iteration"] = record.get("iteration")
    if final is None:
        raise RuntimeError(f"未在 {metrics_file} 中找到最终 bbox/AP。")
    return final


def raw_snapshot(directory: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != MANIFEST
        and path.relative_to(directory).as_posix() not in DERIVED_FILES
    )
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "last_checkpoint": (directory / "last_checkpoint")
        .read_text(encoding="utf-8")
        .strip(),
        "final_metrics": final_bbox_metrics(directory / "metrics.json"),
    }


def validate_run(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise RuntimeError(f"目录不存在：{directory}")
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"{directory} 缺少必需文件：{missing}")
    snapshot = raw_snapshot(directory)
    checkpoint = directory / snapshot["last_checkpoint"]
    if not checkpoint.is_file():
        raise RuntimeError(f"{directory}/last_checkpoint 指向不存在的文件。")
    return snapshot


def manifest_payload(
    spec: dict[str, Any],
    snapshot: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": spec["status"],
        "experiment": {
            "id": spec["id"],
            "name": spec["name"],
            "dataset": spec["dataset"],
            "description": EXPERIMENT_DESCRIPTIONS[spec["id"]],
            "config_file": spec["config"],
        },
        "migration": {
            "source_directory": spec["old"],
            "target_directory": spec["new"],
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "same_filesystem_move": True,
        },
        "execution": {
            "original_command": None,
            "original_command_note": (
                "原命令只打印到终端，log.txt 未捕获 Command Line Args；"
                "配置与恢复行为已从 config.yaml 和 log.txt 还原。"
            ),
            "training_time": spec["training_time"],
        },
        "model": {
            "parameters_million": spec["parameters_million"],
            **spec["parameters"],
        },
        "results": snapshot["final_metrics"],
        "resources": {"peak_memory_mb": spec["peak_memory_mb"]},
        "known_issues": spec["issues"],
        "evidence": {
            "raw_files_immutable": True,
            "raw_file_count": snapshot["file_count"],
            "raw_total_bytes": snapshot["total_bytes"],
            "last_checkpoint": snapshot["last_checkpoint"],
            "metric_source": "metrics.json 最后一条含 bbox/AP 的记录",
            "configuration_source": "config.yaml；冲突时以 log.txt 为准",
            "repository_root_at_migration": str(repository_root),
        },
    }


def preflight(repository_root: Path):
    snapshots = {}
    root_device = repository_root.stat().st_dev
    for spec in RUNS:
        source = repository_root / spec["old"]
        target = repository_root / spec["new"]
        if target.exists():
            raise RuntimeError(f"目标已存在，拒绝迁移：{target}")
        if source.stat().st_dev != root_device:
            raise RuntimeError(f"源与目标不在同一文件系统：{source}")
        snapshots[spec["id"]] = validate_run(source)
    return snapshots


def execute(repository_root: Path) -> None:
    snapshots = preflight(repository_root)
    moved: list[tuple[Path, Path]] = []
    manifests: list[Path] = []
    try:
        for spec in RUNS:
            source = repository_root / spec["old"]
            target = repository_root / spec["new"]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, target)
            moved.append((source, target))

        for spec in RUNS:
            target = repository_root / spec["new"]
            after = validate_run(target)
            if after != snapshots[spec["id"]]:
                raise RuntimeError(f"迁移前后完整性不一致：{spec['id']}")

        for spec in RUNS:
            target = repository_root / spec["new"]
            path = target / MANIFEST
            payload = manifest_payload(
                spec, snapshots[spec["id"]], repository_root
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            manifests.append(path)
    except BaseException:
        for manifest in reversed(manifests):
            if manifest.is_file():
                manifest.unlink()
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                os.rename(target, source)
        raise


def verify(repository_root: Path) -> None:
    for spec in RUNS:
        source = repository_root / spec["old"]
        target = repository_root / spec["new"]
        if source.exists():
            raise RuntimeError(f"旧目录仍存在：{source}")
        snapshot = validate_run(target)
        manifest_path = target / MANIFEST
        if not manifest_path.is_file():
            raise RuntimeError(f"缺少历史清单：{manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2:
            raise RuntimeError(f"运行清单 schema 版本不是 2：{spec['id']}")
        experiment = manifest.get("experiment", {})
        allowed_experiment_fields = {
            "id",
            "name",
            "dataset",
            "description",
            "config_file",
        }
        unexpected = set(experiment) - allowed_experiment_fields
        if unexpected:
            raise RuntimeError(
                f"运行清单含意外实验字段：{spec['id']} {sorted(unexpected)}"
            )
        if experiment.get("description") != EXPERIMENT_DESCRIPTIONS[spec["id"]]:
            raise RuntimeError(f"运行清单说明不一致：{spec['id']}")
        evidence = manifest["evidence"]
        if snapshot["file_count"] != evidence["raw_file_count"]:
            raise RuntimeError(f"原始文件数量变化：{spec['id']}")
        if snapshot["total_bytes"] != evidence["raw_total_bytes"]:
            raise RuntimeError(f"原始文件字节数变化：{spec['id']}")
        if snapshot["last_checkpoint"] != evidence["last_checkpoint"]:
            raise RuntimeError(f"last_checkpoint 变化：{spec['id']}")
        if snapshot["final_metrics"] != manifest["results"]:
            raise RuntimeError(f"最终指标变化：{spec['id']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="执行迁移")
    action.add_argument("--verify", action="store_true", help="验证迁移结果")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    try:
        if args.verify:
            verify(root)
            print("历史实验迁移验证通过。")
        elif args.execute:
            execute(root)
            verify(root)
            print("历史实验迁移完成且验证通过。")
        else:
            snapshots = preflight(root)
            print("Dry-run 通过；未移动任何目录。")
            for spec in RUNS:
                snapshot = snapshots[spec["id"]]
                print(
                    f"{spec['old']} -> {spec['new']} | "
                    f"files={snapshot['file_count']} "
                    f"bytes={snapshot['total_bytes']} "
                    f"AP={snapshot['final_metrics']['AP']}"
                )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
