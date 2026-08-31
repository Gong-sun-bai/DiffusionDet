#!/usr/bin/env python3
"""Evaluate selected DynamicHead stages of an existing checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detectron2.config import get_cfg

from diffusiondet import add_diffusiondet_config, add_mobilenetv4_config
from diffusiondet.util.model_ema import add_model_ema_configs
from experiment_manager import metadata_from_cfg, resolve_run_dir


def load_context(config: Path):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(str(config))
    metadata = metadata_from_cfg(cfg)
    return cfg, resolve_run_dir(ROOT, metadata)


def newest_summary(
    evaluations: Path,
    previous,
    *,
    stage: int,
    sample_step: int,
    ensemble_mode: str,
    seed: int,
):
    candidates = {
        path.parent: path.stat().st_mtime_ns
        for path in evaluations.glob("*/result_summary.json")
    }
    created = []
    for path in candidates:
        if path in previous:
            continue
        config_path = path / "config.yaml"
        if not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        diffusion = config["MODEL"]["DiffusionDet"]
        if (
            int(diffusion["INFERENCE_HEAD_STAGE"]) == stage
            and int(diffusion["SAMPLE_STEP"]) == sample_step
            and str(diffusion["ENSEMBLE_MODE"]) == ensemble_mode
            and int(config["SEED"]) == seed
        ):
            created.append(path)
    if not created:
        raise RuntimeError("evaluation completed without a new result_summary.json")
    directory = max(created, key=lambda path: candidates[path])
    payload = json.loads((directory / "result_summary.json").read_text(encoding="utf-8"))
    return directory, payload.get("results", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--ensemble-mode", default="legacy_nonfinal")
    parser.add_argument("--seed", type=int, default=40244023)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = args.config.resolve()
    weights = args.weights.resolve()
    cfg, run_dir = load_context(config)
    evaluations = run_dir / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    lock_handle = (evaluations / ".head-stage-evaluation.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another head-stage evaluation is already running") from error
    summary = []
    for stage in args.stages:
        command = [
            sys.executable,
            str(ROOT / "train_net.py"),
            "--num-gpus",
            "1",
            "--config-file",
            str(config),
            "--eval-only",
            "MODEL.WEIGHTS",
            str(weights),
            "MODEL.DiffusionDet.INFERENCE_HEAD_STAGE",
            str(stage),
            "MODEL.DiffusionDet.SAMPLE_STEP",
            str(args.sample_step),
            "MODEL.DiffusionDet.ENSEMBLE_MODE",
            args.ensemble_mode,
            "SEED",
            str(args.seed),
        ]
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        previous = {path.parent for path in evaluations.glob("*/result_summary.json")}
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=None if args.verbose else subprocess.STDOUT,
        )
        directory, results = newest_summary(
            evaluations,
            previous,
            stage=stage,
            sample_step=args.sample_step,
            ensemble_mode=args.ensemble_mode,
            seed=args.seed,
        )
        bbox = results.get("bbox", {})
        summary.append(
            {
                "stage": stage,
                "seed": args.seed,
                "evaluation": str(directory.relative_to(ROOT)),
                "AP": bbox.get("AP"),
                "AP50": bbox.get("AP50"),
                "AP75": bbox.get("AP75"),
                "APs": bbox.get("APs"),
            }
        )
    if not args.dry_run:
        encoded = json.dumps(summary, ensure_ascii=False, indent=2)
        print(encoded)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
