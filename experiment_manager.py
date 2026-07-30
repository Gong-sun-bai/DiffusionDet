"""Strict, dependency-free experiment lifecycle helpers.

This module intentionally uses only the Python standard library so its safety
rules can be tested before the model environment is restored.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


EXPERIMENT_ID_RE = re.compile(r"^[a-z][a-z0-9]*-\d{3}$")
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DATASET_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CHANGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*:\s*.+\s+->\s+.+$")
ALLOWED_KINDS = {"historical", "baseline", "ablation", "smoke"}
PREDICTION_FILENAME = "instances_predictions.pth"
MANIFEST_FILENAME = "experiment_manifest.json"
SUMMARY_FILENAME = "result_summary.json"
SCHEMA_VERSION = 1


class ExperimentError(ValueError):
    """Raised when an experiment would violate the repository rules."""


@dataclass(frozen=True)
class ExperimentMetadata:
    experiment_id: str
    name: str
    dataset: str
    track: str
    kind: str
    baseline: str
    purpose: str
    hypothesis: str
    changes: tuple[str, ...]
    output_root: str


@dataclass(frozen=True)
class ExperimentContext:
    metadata: ExperimentMetadata
    repository_root: Path
    run_dir: Path
    output_dir: Path
    mode: str
    config_file: Path
    command: tuple[str, ...]
    weights: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(value: Optional[datetime] = None) -> str:
    value = value or utc_now()
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _get(node: Any, key: str, default: Any = None) -> Any:
    current = node
    for part in key.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def metadata_from_cfg(cfg: Any) -> ExperimentMetadata:
    changes = _get(cfg, "EXPERIMENT.CHANGES", ())
    return ExperimentMetadata(
        experiment_id=str(_get(cfg, "EXPERIMENT.ID", "")).strip(),
        name=str(_get(cfg, "EXPERIMENT.NAME", "")).strip(),
        dataset=str(_get(cfg, "EXPERIMENT.DATASET", "")).strip(),
        track=str(_get(cfg, "EXPERIMENT.TRACK", "")).strip(),
        kind=str(_get(cfg, "EXPERIMENT.KIND", "")).strip(),
        baseline=str(_get(cfg, "EXPERIMENT.BASELINE", "")).strip(),
        purpose=str(_get(cfg, "EXPERIMENT.PURPOSE", "")).strip(),
        hypothesis=str(_get(cfg, "EXPERIMENT.HYPOTHESIS", "")).strip(),
        changes=tuple(str(item).strip() for item in changes),
        output_root=str(_get(cfg, "EXPERIMENT.OUTPUT_ROOT", "./runs")).strip(),
    )


def validate_slug(name: str) -> None:
    if not NAME_RE.fullmatch(name):
        raise ExperimentError(
            "EXPERIMENT.NAME 只能包含小写字母、数字和单连字符分隔的词。"
        )


def validate_experiment_id(experiment_id: str) -> None:
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ExperimentError("EXPERIMENT.ID 必须符合 <数据集短名>-<三位序号>。")


def validate_metadata(
    metadata: ExperimentMetadata,
    *,
    seed: int,
    max_iter: int,
    steps: Iterable[int],
) -> None:
    validate_experiment_id(metadata.experiment_id)
    validate_slug(metadata.name)
    if not DATASET_RE.fullmatch(metadata.dataset):
        raise ExperimentError("EXPERIMENT.DATASET 必须是小写 snake_case 名称。")
    expected_prefix = metadata.dataset.split("_", 1)[0]
    if metadata.experiment_id.split("-", 1)[0] != expected_prefix:
        raise ExperimentError(
            "EXPERIMENT.ID 前缀必须与 EXPERIMENT.DATASET 的短名一致。"
        )
    if not metadata.track:
        raise ExperimentError("EXPERIMENT.TRACK 不能为空。")
    if metadata.kind not in ALLOWED_KINDS:
        raise ExperimentError(
            "EXPERIMENT.KIND 只能为 historical、baseline、ablation 或 smoke。"
        )
    if not metadata.purpose:
        raise ExperimentError("EXPERIMENT.PURPOSE 不能为空。")
    if not metadata.hypothesis:
        raise ExperimentError("EXPERIMENT.HYPOTHESIS 不能为空。")
    if not metadata.output_root:
        raise ExperimentError("EXPERIMENT.OUTPUT_ROOT 不能为空。")
    if metadata.kind != "historical" and seed < 0:
        raise ExperimentError("非历史实验必须设置固定的非负 SEED。")
    invalid_steps = [int(step) for step in steps if int(step) >= int(max_iter)]
    if metadata.kind != "historical" and invalid_steps:
        raise ExperimentError(
            f"所有 SOLVER.STEPS 必须小于 SOLVER.MAX_ITER；无效值：{invalid_steps}"
        )
    if metadata.kind == "ablation":
        if not metadata.baseline:
            raise ExperimentError("消融实验必须填写 EXPERIMENT.BASELINE。")
        if not metadata.changes:
            raise ExperimentError("消融实验必须填写 EXPERIMENT.CHANGES。")
        malformed = [item for item in metadata.changes if not CHANGE_RE.fullmatch(item)]
        if malformed:
            raise ExperimentError(
                "EXPERIMENT.CHANGES 必须使用“配置键: 旧值 -> 新值”格式："
                + repr(malformed)
            )


def option_was_explicit(options: Sequence[str], key: str) -> bool:
    return any(options[index] == key for index in range(0, len(options), 2))


def validate_weights(weights: str) -> None:
    if Path(weights).name == PREDICTION_FILENAME:
        raise ExperimentError(
            f"{PREDICTION_FILENAME} 是预测结果，不是模型权重；请使用 model_final.pth。"
        )


def resolve_output_root(repository_root: Path, output_root: str) -> Path:
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = repository_root / root
    return root.resolve()


def resolve_run_dir(repository_root: Path, metadata: ExperimentMetadata) -> Path:
    output_root = resolve_output_root(repository_root, metadata.output_root)
    return output_root / metadata.dataset / (
        f"{metadata.experiment_id}__{metadata.name}"
    )


def configure_experiment(
    cfg: Any,
    args: Any,
    repository_root: Path,
    *,
    now: Optional[datetime] = None,
) -> ExperimentContext:
    repository_root = Path(repository_root).resolve()
    metadata = metadata_from_cfg(cfg)
    validate_metadata(
        metadata,
        seed=int(_get(cfg, "SEED", -1)),
        max_iter=int(_get(cfg, "SOLVER.MAX_ITER", 0)),
        steps=_get(cfg, "SOLVER.STEPS", ()),
    )

    eval_only = bool(getattr(args, "eval_only", False))
    resume = bool(getattr(args, "resume", False))
    options = tuple(getattr(args, "opts", ()) or ())
    if eval_only and resume:
        raise ExperimentError("--eval-only 与 --resume 不能同时使用。")

    mode = "evaluation" if eval_only else ("resume" if resume else "train")
    if metadata.kind == "historical" and mode != "evaluation":
        raise ExperimentError("历史配置禁止重新训练，只允许显式权重评估。")

    weights = str(_get(cfg, "MODEL.WEIGHTS", "") or "")
    validate_weights(weights)
    if eval_only:
        if not option_was_explicit(options, "MODEL.WEIGHTS"):
            raise ExperimentError(
                "评估必须在命令行显式指定 MODEL.WEIGHTS，避免误用冻结配置。"
            )
        if Path(weights).name != "model_final.pth":
            raise ExperimentError("评估权重必须明确指向 model_final.pth。")

    run_dir = resolve_run_dir(repository_root, metadata)
    output_dir = (
        run_dir / "evaluations" / utc_timestamp(now)
        if mode == "evaluation"
        else run_dir
    )
    config_file = Path(getattr(args, "config_file", ""))
    if not config_file.is_absolute():
        config_file = (repository_root / config_file).resolve()

    return ExperimentContext(
        metadata=metadata,
        repository_root=repository_root,
        run_dir=run_dir,
        output_dir=output_dir,
        mode=mode,
        config_file=config_file,
        command=tuple(sys.argv),
        weights=weights,
    )


def validate_output_state(context: ExperimentContext) -> None:
    if context.mode == "resume":
        marker = context.run_dir / "last_checkpoint"
        if not marker.is_file():
            raise ExperimentError(
                f"--resume 要求存在有效的 {marker.relative_to(context.repository_root)}。"
            )
        checkpoint_name = marker.read_text(encoding="utf-8").strip()
        checkpoint = context.run_dir / checkpoint_name
        if not checkpoint_name or not checkpoint.is_file():
            raise ExperimentError(
                f"last_checkpoint 指向的检查点不存在：{checkpoint_name!r}"
            )
        return

    if context.mode == "evaluation":
        if not context.run_dir.is_dir():
            raise ExperimentError(f"待评估实验目录不存在：{context.run_dir}")
        weights = Path(context.weights).expanduser()
        if not weights.is_absolute():
            weights = context.repository_root / weights
        if not weights.is_file():
            raise ExperimentError(f"评估权重不存在：{weights}")
        if context.output_dir.exists():
            raise ExperimentError(f"评估输出目录已存在：{context.output_dir}")
        return

    if context.output_dir.exists() and any(context.output_dir.iterdir()):
        raise ExperimentError(
            f"新训练拒绝复用非空目录：{context.output_dir}；请更换实验 ID/名称。"
        )


def _git_state(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _environment_summary() -> dict[str, Any]:
    packages = {}
    for package in ("torch", "torchvision", "detectron2", "fvcore", "timm"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            module = sys.modules.get(package)
            packages[package] = (
                getattr(module, "__version__", None) if module is not None else None
            )
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "packages": packages,
    }


def _key_parameters(cfg: Any) -> dict[str, Any]:
    keys = (
        "SEED",
        "MODEL.WEIGHTS",
        "MODEL.BACKBONE.NAME",
        "MODEL.RESNETS.DEPTH",
        "MODEL.FPN.OUT_CHANNELS",
        "MODEL.DiffusionDet.HIDDEN_DIM",
        "MODEL.DiffusionDet.NUM_PROPOSALS",
        "MODEL.DiffusionDet.SAMPLE_STEP",
        "SOLVER.IMS_PER_BATCH",
        "SOLVER.BASE_LR",
        "SOLVER.MAX_ITER",
        "SOLVER.STEPS",
        "SOLVER.WARMUP_ITERS",
        "SOLVER.CHECKPOINT_PERIOD",
        "SOLVER.AMP.ENABLED",
        "DATALOADER.NUM_WORKERS",
        "INPUT.MIN_SIZE_TRAIN",
        "INPUT.MAX_SIZE_TRAIN",
        "INPUT.MIN_SIZE_TEST",
        "INPUT.MAX_SIZE_TEST",
        "DATASETS.TRAIN",
        "DATASETS.TEST",
    )
    values: dict[str, Any] = {}
    for key in keys:
        value = _get(cfg, key)
        if isinstance(value, tuple):
            value = list(value)
        values[key] = value
    return values


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_experiment(context: ExperimentContext, cfg: Any) -> dict[str, Any]:
    validate_output_state(context)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = context.output_dir / MANIFEST_FILENAME
    execution_history = []
    if context.mode == "resume" and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        execution_history = list(previous.get("execution_history", ()))
        if previous.get("execution"):
            execution_history.append(previous["execution"])
    started_at = utc_now().isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "experiment": {
            "id": context.metadata.experiment_id,
            "name": context.metadata.name,
            "dataset": context.metadata.dataset,
            "track": context.metadata.track,
            "kind": context.metadata.kind,
            "baseline": context.metadata.baseline or None,
            "purpose": context.metadata.purpose,
            "hypothesis": context.metadata.hypothesis,
            "changes": list(context.metadata.changes),
        },
        "execution": {
            "mode": context.mode,
            "started_at": started_at,
            "finished_at": None,
            "config_file": str(context.config_file),
            "output_dir": str(context.output_dir),
            "command": list(context.command),
            "git": _git_state(context.repository_root),
            "environment": _environment_summary(),
        },
        "parameters": _key_parameters(cfg),
        "execution_history": execution_history,
        "error": None,
    }
    atomic_write_json(manifest_path, payload)
    return payload


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def finish_experiment(
    context: ExperimentContext,
    *,
    results: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    manifest_path = context.output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = "failed" if error is not None else "completed"
    payload["execution"]["finished_at"] = utc_now().isoformat()
    payload["error"] = (
        {"type": type(error).__name__, "message": str(error)}
        if error is not None
        else None
    )
    atomic_write_json(manifest_path, payload)
    if error is None:
        atomic_write_json(
            context.output_dir / SUMMARY_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": context.metadata.experiment_id,
                "status": "completed",
                "finished_at": payload["execution"]["finished_at"],
                "results": _json_safe(results),
            },
        )
