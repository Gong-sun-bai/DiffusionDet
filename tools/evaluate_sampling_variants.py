#!/usr/bin/env python3
"""Evaluate DDIM step/ensemble variants for an existing experiment."""

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


DEFAULT_VARIANTS = (
    (1, "legacy_nonfinal"),
    (1, "final_only"),
    (1, "all_steps"),
    (2, "legacy_nonfinal"),
    (2, "final_only"),
    (2, "all_steps"),
    (4, "legacy_nonfinal"),
    (4, "final_only"),
    (4, "all_steps"),
)


def parse_variant(value: str):
    try:
        steps_text, mode = value.split(":", 1)
        steps = int(steps_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "variant must use STEPS:MODE syntax"
        ) from error
    if steps not in {1, 2, 4} or mode not in {
        "legacy_nonfinal",
        "final_only",
        "all_steps",
    }:
        raise argparse.ArgumentTypeError(
            "variant must use steps 1/2/4 and a supported ensemble mode"
        )
    return steps, mode


def run_dir_for(config: Path):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_mobilenetv4_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(str(config))
    return resolve_run_dir(ROOT, metadata_from_cfg(cfg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--variants",
        type=parse_variant,
        nargs="*",
        help="Optional STEPS:MODE subset; default is the full 3-by-3 matrix.",
    )
    parser.add_argument("--seed", type=int, default=40244023)
    parser.add_argument(
        "--extra-seeds",
        type=int,
        nargs="*",
        default=[40244024, 40244025],
        help="Re-evaluate the best matrix variant with these additional seeds.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    weights = args.weights.resolve()
    evaluations = run_dir_for(config) / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    lock_handle = (evaluations / ".sampling-variant-evaluation.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            "another sampling-variant evaluation is already running"
        ) from error
    collected = []
    def evaluate(steps, mode, seed):
        before = set(evaluations.glob("*/result_summary.json"))
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
            "MODEL.DiffusionDet.SAMPLE_STEP",
            str(steps),
            "MODEL.DiffusionDet.ENSEMBLE_MODE",
            mode,
            "SEED",
            str(seed),
        ]
        print(" ".join(command), flush=True)
        if args.dry_run:
            return None
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=None if args.verbose else subprocess.STDOUT,
        )
        created = []
        for summary_path in set(evaluations.glob("*/result_summary.json")) - before:
            config_path = summary_path.parent / "config.yaml"
            if not config_path.is_file():
                continue
            config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            diffusion = config_payload["MODEL"]["DiffusionDet"]
            if (
                int(diffusion["SAMPLE_STEP"]) == steps
                and str(diffusion["ENSEMBLE_MODE"]) == mode
                and int(config_payload["SEED"]) == seed
            ):
                created.append(summary_path)
        if len(created) != 1:
            raise RuntimeError(
                "expected one matching new evaluation summary, "
                f"got {len(created)}"
            )
        summary_path = created[0]
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        bbox = payload.get("results", {}).get("bbox", {})
        return {
            "steps": steps,
            "ensemble_mode": mode,
            "seed": seed,
            "evaluation": str(summary_path.parent.relative_to(ROOT)),
            **{
                key: bbox.get(key)
                for key in ("AP", "AP50", "AP75", "APs", "APm", "APl")
            },
        }

    variants = args.variants or DEFAULT_VARIANTS
    for steps, mode in variants:
        result = evaluate(steps, mode, args.seed)
        if result is not None:
            collected.append(result)
    if not args.dry_run:
        best = max(
            collected,
            key=lambda item: (item["AP"], item["AP75"], item["APs"]),
        )
        for seed in args.extra_seeds:
            if seed == args.seed:
                continue
            collected.append(evaluate(best["steps"], best["ensemble_mode"], seed))
    if not args.dry_run:
        encoded = json.dumps(collected, ensure_ascii=False, indent=2)
        print(encoded)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
