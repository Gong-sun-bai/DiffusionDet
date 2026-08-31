"""Strict, dependency-free experiment lifecycle helpers.

This module intentionally uses only the Python standard library so its safety
rules can be tested before the model environment is restored.
"""

from __future__ import annotations

import json
import hashlib
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


EXPERIMENT_ID_RE = re.compile(
    r"^(?P<prefix>[a-z][a-z0-9]*)(?P<test>-test)?-\d{3}$"
)
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DATASET_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PREDICTION_FILENAME = "instances_predictions.pth"
MODEL_WEIGHT_FILENAMES = {
    "model_best.pth",
    "model_final.pth",
    "model_latest.pth",
}
MANIFEST_FILENAME = "experiment_manifest.json"
SUMMARY_FILENAME = "result_summary.json"
MANIFEST_SCHEMA_VERSION = 2
SUMMARY_SCHEMA_VERSION = 1


class ExperimentError(ValueError):
    """Raised when an experiment would violate the repository rules."""


@dataclass(frozen=True)
class ExperimentMetadata:
    experiment_id: str
    name: str
    dataset: str
    description: str
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
    run_until_iter: Optional[int]


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
    return ExperimentMetadata(
        experiment_id=str(_get(cfg, "EXPERIMENT.ID", "")).strip(),
        name=str(_get(cfg, "EXPERIMENT.NAME", "")).strip(),
        dataset=str(_get(cfg, "EXPERIMENT.DATASET", "")).strip(),
        description=str(_get(cfg, "EXPERIMENT.DESCRIPTION", "") or "").strip(),
        output_root=str(_get(cfg, "EXPERIMENT.OUTPUT_ROOT", "./runs")).strip(),
    )


def validate_slug(name: str) -> None:
    if not NAME_RE.fullmatch(name):
        raise ExperimentError(
            "EXPERIMENT.NAME 只能包含小写字母、数字和单连字符分隔的词。"
        )


def validate_experiment_id(experiment_id: str) -> None:
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ExperimentError(
            "EXPERIMENT.ID 必须符合正式实验 <数据集短名>-<三位序号>，"
            "或测试实验 <数据集短名>-test-<三位序号>。"
        )


def validate_metadata(
    metadata: ExperimentMetadata,
) -> None:
    validate_experiment_id(metadata.experiment_id)
    validate_slug(metadata.name)
    if not DATASET_RE.fullmatch(metadata.dataset):
        raise ExperimentError("EXPERIMENT.DATASET 必须是小写 snake_case 名称。")
    id_match = EXPERIMENT_ID_RE.fullmatch(metadata.experiment_id)
    expected_prefix = metadata.dataset.split("_", 1)[0]
    if id_match is None or id_match.group("prefix") != expected_prefix:
        raise ExperimentError(
            "EXPERIMENT.ID 前缀必须与 EXPERIMENT.DATASET 的短名一致。"
        )
    if not metadata.output_root:
        raise ExperimentError("EXPERIMENT.OUTPUT_ROOT 不能为空。")


def validate_training_parameters(
    *,
    seed: int,
    max_iter: int,
    steps: Iterable[int],
) -> None:
    if seed < 0:
        raise ExperimentError("训练或续训必须设置固定的非负 SEED。")
    invalid_steps = [int(step) for step in steps if int(step) >= int(max_iter)]
    if invalid_steps:
        raise ExperimentError(
            f"所有 SOLVER.STEPS 必须小于 SOLVER.MAX_ITER；无效值：{invalid_steps}"
        )


def option_was_explicit(options: Sequence[str], key: str) -> bool:
    return any(options[index] == key for index in range(0, len(options), 2))


def validate_weights(weights: str) -> None:
    if Path(weights).name == PREDICTION_FILENAME:
        raise ExperimentError(
            f"{PREDICTION_FILENAME} 是预测结果，不是模型权重；"
            "请使用 model_final.pth、model_latest.pth 或 model_best.pth。"
        )


def validate_checkpoint_policy(cfg: Any) -> None:
    retention = str(_get(cfg, "SOLVER.CHECKPOINT_RETENTION", "all"))
    if retention not in {"all", "latest"}:
        raise ExperimentError(
            "SOLVER.CHECKPOINT_RETENTION 只能为 all 或 latest。"
        )
    if retention == "latest" and int(
        _get(cfg, "SOLVER.CHECKPOINT_PERIOD", 0)
    ) <= 0:
        raise ExperimentError(
            "SOLVER.CHECKPOINT_RETENTION=latest 时 "
            "SOLVER.CHECKPOINT_PERIOD 必须大于 0。"
        )

    if not bool(_get(cfg, "TEST.BEST_CHECKPOINT.ENABLED", False)):
        return

    eval_period = int(_get(cfg, "TEST.EVAL_PERIOD", 0))
    if eval_period <= 0:
        raise ExperimentError(
            "启用 TEST.BEST_CHECKPOINT 时 TEST.EVAL_PERIOD 必须大于 0。"
        )
    metric = str(_get(cfg, "TEST.BEST_CHECKPOINT.METRIC", "")).strip()
    if not metric:
        raise ExperimentError("TEST.BEST_CHECKPOINT.METRIC 不能为空。")
    mode = str(_get(cfg, "TEST.BEST_CHECKPOINT.MODE", ""))
    if mode not in {"max", "min"}:
        raise ExperimentError(
            "TEST.BEST_CHECKPOINT.MODE 只能为 max 或 min。"
        )
    if not tuple(_get(cfg, "DATASETS.TEST", ()) or ()):
        raise ExperimentError(
            "启用 TEST.BEST_CHECKPOINT 时 DATASETS.TEST 不能为空。"
        )


def validate_training_plot_policy(cfg: Any) -> None:
    if not bool(_get(cfg, "TRAINING_PLOTS.ENABLED", True)):
        return
    if int(_get(cfg, "TRAINING_PLOTS.PERIOD", 200)) <= 0:
        raise ExperimentError(
            "启用 TRAINING_PLOTS 时 TRAINING_PLOTS.PERIOD 必须大于 0。"
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


def existing_training_result(run_dir: Path) -> Optional[Path]:
    """Return a canonical non-empty run directory, if one already exists."""
    run_dir = Path(run_dir)
    if run_dir.is_dir() and any(run_dir.iterdir()):
        return run_dir.resolve()
    return None


def configure_experiment(
    cfg: Any,
    args: Any,
    repository_root: Path,
    *,
    now: Optional[datetime] = None,
) -> ExperimentContext:
    repository_root = Path(repository_root).resolve()
    metadata = metadata_from_cfg(cfg)
    eval_only = bool(getattr(args, "eval_only", False))
    resume = bool(getattr(args, "resume", False))
    options = tuple(getattr(args, "opts", ()) or ())
    run_until_iter = getattr(args, "run_until_iter", None)
    if run_until_iter is not None:
        run_until_iter = int(run_until_iter)
    if eval_only and resume:
        raise ExperimentError("--eval-only 与 --resume 不能同时使用。")
    if eval_only and run_until_iter is not None:
        raise ExperimentError("--eval-only 不能与 --run-until-iter 同时使用。")

    mode = "evaluation" if eval_only else ("resume" if resume else "train")
    validate_metadata(metadata)
    run_dir = resolve_run_dir(repository_root, metadata)
    repeated_training = (
        mode == "train" and existing_training_result(run_dir) is not None
    )
    if mode != "evaluation" and not repeated_training:
        max_iter = int(_get(cfg, "SOLVER.MAX_ITER", 0))
        validate_training_parameters(
            seed=int(_get(cfg, "SEED", -1)),
            max_iter=max_iter,
            steps=_get(cfg, "SOLVER.STEPS", ()),
        )
        if run_until_iter is not None and not 0 < run_until_iter <= max_iter:
            raise ExperimentError(
                "--run-until-iter 必须大于 0 且不超过 SOLVER.MAX_ITER。"
            )
    validate_checkpoint_policy(cfg)
    validate_training_plot_policy(cfg)

    weights = str(_get(cfg, "MODEL.WEIGHTS", "") or "")
    validate_weights(weights)
    if eval_only:
        if not option_was_explicit(options, "MODEL.WEIGHTS"):
            raise ExperimentError(
                "评估必须在命令行显式指定 MODEL.WEIGHTS，避免误用冻结配置。"
            )
        if Path(weights).name not in MODEL_WEIGHT_FILENAMES:
            allowed = "、".join(sorted(MODEL_WEIGHT_FILENAMES))
            raise ExperimentError(f"评估权重文件名只能为：{allowed}。")

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
        run_until_iter=run_until_iter,
    )


def validate_output_state(context: ExperimentContext) -> Optional[Path]:
    """Validate the target state and return an existing training result.

    A plain training invocation is idempotent: when its canonical, non-empty
    run directory already exists, the caller receives that path and can exit
    normally without touching any historical artifact.  Resume and evaluation
    retain their stricter validation rules.
    """
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
        return None

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
        return None

    if context.output_dir.exists() and not context.output_dir.is_dir():
        raise ExperimentError(f"实验输出路径不是目录：{context.output_dir}")
    existing_run_dir = existing_training_result(context.output_dir)
    if existing_run_dir is not None:
        return existing_run_dir
    return None


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


def config_sha256(cfg: Any) -> str:
    if hasattr(cfg, "dump"):
        serialized = cfg.dump()
    else:
        serialized = json.dumps(
            _json_safe(cfg),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _key_parameters(cfg: Any) -> dict[str, Any]:
    keys = (
        "SEED",
        "MODEL.WEIGHTS",
        "MODEL.BACKBONE.NAME",
        "MODEL.RESNETS.DEPTH",
        "MODEL.FPN.OUT_CHANNELS",
        "MODEL.DiffusionDet.HIDDEN_DIM",
        "MODEL.DiffusionDet.HEAD_SHARING",
        "MODEL.DiffusionDet.NUM_PROPOSALS",
        "MODEL.DiffusionDet.SAMPLE_STEP",
        "MODEL.DiffusionDet.ENSEMBLE_MODE",
        "MODEL.DiffusionDet.INFERENCE_HEAD_STAGE",
        "MODEL.DiffusionDet.DISTILLATION.ENABLED",
        "MODEL.TIMM.NAME",
        "MODEL.TIMM.PRETRAINED",
        "SOLVER.IMS_PER_BATCH",
        "SOLVER.BASE_LR",
        "SOLVER.MAX_ITER",
        "SOLVER.STEPS",
        "SOLVER.WARMUP_ITERS",
        "SOLVER.CHECKPOINT_PERIOD",
        "SOLVER.CHECKPOINT_RETENTION",
        "SOLVER.AMP.ENABLED",
        "TEST.EVAL_PERIOD",
        "TEST.BEST_CHECKPOINT.ENABLED",
        "TEST.BEST_CHECKPOINT.METRIC",
        "TEST.BEST_CHECKPOINT.MODE",
        "TRAINING_PLOTS.ENABLED",
        "TRAINING_PLOTS.PERIOD",
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
    existing_run_dir = validate_output_state(context)
    if existing_run_dir is not None:
        raise ExperimentError(
            f"实验结果目录已经存在，不能重新初始化：{existing_run_dir}"
        )
    context.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = context.output_dir / MANIFEST_FILENAME
    execution_history = []
    screening_history = []
    if context.mode == "resume" and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        execution_history = list(previous.get("execution_history", ()))
        screening_history = list(previous.get("screening_history", ()))
        if previous.get("execution"):
            execution_history.append(previous["execution"])
    started_at = utc_now().isoformat()
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "experiment": {
            "id": context.metadata.experiment_id,
            "name": context.metadata.name,
            "dataset": context.metadata.dataset,
            "description": context.metadata.description or None,
        },
        "execution": {
            "mode": context.mode,
            "started_at": started_at,
            "finished_at": None,
            "config_file": str(context.config_file),
            "output_dir": str(context.output_dir),
            "command": list(context.command),
            "run_until_iter": context.run_until_iter,
            "config_sha256": config_sha256(cfg),
            "git": _git_state(context.repository_root),
            "environment": _environment_summary(),
        },
        "parameters": _key_parameters(cfg),
        "execution_history": execution_history,
        "screening_history": screening_history,
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
    status: Optional[str] = None,
) -> None:
    manifest_path = context.output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if status is not None and status not in {
        "paused",
        "promoted",
        "screened_out",
        "completed",
        "failed",
    }:
        raise ValueError(f"unsupported experiment status: {status}")
    payload["status"] = (
        "failed" if error is not None else (status or "completed")
    )
    payload["execution"]["finished_at"] = utc_now().isoformat()
    payload["error"] = (
        {"type": type(error).__name__, "message": str(error)}
        if error is not None
        else None
    )
    atomic_write_json(manifest_path, payload)
    if error is None and payload["status"] == "completed":
        atomic_write_json(
            context.output_dir / SUMMARY_FILENAME,
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "experiment_id": context.metadata.experiment_id,
                "status": "completed",
                "finished_at": payload["execution"]["finished_at"],
                "results": _json_safe(results),
            },
        )
