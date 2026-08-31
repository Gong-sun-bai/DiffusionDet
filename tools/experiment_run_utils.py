"""Shared read/validation helpers for experiment orchestration tools."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detectron2.config import get_cfg  # noqa: E402

from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config  # noqa: E402
from diffusiondet.util.model_ema import add_model_ema_configs  # noqa: E402
from experiment_manager import config_sha256, configure_experiment  # noqa: E402


@dataclass(frozen=True)
class LoadedExperiment:
    config: Path
    run_dir: Path
    config_hash: str
    max_iter: int


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment(config_path: Path) -> LoadedExperiment:
    config_path = config_path.resolve()
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(str(config_path))
    args = SimpleNamespace(
        config_file=str(config_path),
        eval_only=False,
        resume=False,
        run_until_iter=None,
        opts=[],
    )
    context = configure_experiment(cfg, args, ROOT)
    cfg.OUTPUT_DIR = str(context.run_dir)
    cfg.freeze()
    return LoadedExperiment(
        config=config_path,
        run_dir=context.run_dir,
        config_hash=config_sha256(cfg),
        max_iter=int(cfg.SOLVER.MAX_ITER),
    )


def action_for(experiment: LoadedExperiment) -> str:
    run_dir = experiment.run_dir
    manifest_path = run_dir / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        recorded_hash = manifest.get("execution", {}).get("config_sha256")
        if recorded_hash != experiment.config_hash:
            raise RuntimeError(
                f"config hash mismatch for {run_dir}: "
                f"recorded={recorded_hash!r}, current={experiment.config_hash!r}"
            )
        if (
            manifest.get("status") == "completed"
            and (run_dir / "result_summary.json").is_file()
        ):
            return "skip"
    if run_dir.is_dir() and any(run_dir.iterdir()):
        marker = run_dir / "last_checkpoint"
        if marker.is_file():
            checkpoint_name = marker.read_text(encoding="utf-8").strip()
            if checkpoint_name and (run_dir / checkpoint_name).is_file():
                return "resume"
        raise RuntimeError(f"nonempty run has no valid resumable checkpoint: {run_dir}")
    return "train"

