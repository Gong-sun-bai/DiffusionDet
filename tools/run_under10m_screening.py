#!/usr/bin/env python3
"""Run under-10M candidates serially to an absolute screening rung."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.experiment_run_utils import action_for, load_experiment


CANDIDATES = {
    "sar-007": "configs/experiments/sar_ship/sar-007-under10m-control.yaml",
    "sar-008": "configs/experiments/sar_ship/sar-008-under10m-shared-h64.yaml",
    "sar-009": "configs/experiments/sar_ship/sar-009-under10m-shared-h128.yaml",
    "sar-010": "configs/experiments/sar_ship/sar-010-under10m-heavy-h128.yaml",
    "sar-011": "configs/experiments/sar_ship/sar-011-under10m-timm-mnv4-scratch.yaml",
    "sar-012": "configs/experiments/sar_ship/sar-012-under10m-timm-mnv4-pretrained.yaml",
    "sar-013": "configs/experiments/sar_ship/sar-013-under10m-repvit-pretrained.yaml",
}
RUNGS = (200, 5000, 20000, 50000)


def reached_rung(experiment, rung):
    manifest_path = experiment.run_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("execution", {}).get("run_until_iter")
    return (
        manifest.get("status") in {"paused", "promoted", "screened_out", "completed"}
        and isinstance(recorded, int)
        and recorded >= rung
    )


def run_queue(args):
    lock_path = ROOT / "runs/under10m_screening" / f".rung-{args.rung}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another screening queue is already running for rung {args.rung}"
            ) from error

        selected = args.ids or sorted(CANDIDATES)
        for experiment_id in selected:
            experiment = load_experiment(ROOT / CANDIDATES[experiment_id])
            manifest_path = experiment.run_dir / "experiment_manifest.json"
            if manifest_path.is_file() and not args.ids:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "screened_out":
                    print(f"[skip-screened-out] {experiment_id}")
                    continue
            if reached_rung(experiment, args.rung):
                print(f"[skip-rung] {experiment_id} already reached {args.rung}")
                continue
            action = action_for(experiment)
            if action == "skip":
                print(f"[skip-completed] {experiment_id}")
                continue
            command = [
                sys.executable,
                str(ROOT / "train_net.py"),
                "--num-gpus",
                "1",
                "--config-file",
                str(experiment.config),
                "--run-until-iter",
                str(args.rung),
            ]
            if action == "resume":
                command.append("--resume")
            print(f"[{action}] {' '.join(command)}", flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, cwd=ROOT, check=True)
            if not reached_rung(experiment, args.rung):
                raise RuntimeError(
                    f"{experiment_id} returned without paused evidence for rung "
                    f"{args.rung}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", type=int, choices=RUNGS, required=True)
    parser.add_argument("--ids", nargs="+", choices=sorted(CANDIDATES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run the queue in an independent session and append output to a log",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="detached log path (default: runs/under10m_screening/logs/rung-N.log)",
    )
    args = parser.parse_args()
    if not args.detach:
        run_queue(args)
        return

    if args.dry_run:
        raise ValueError("--detach and --dry-run cannot be combined")
    log_path = args.log_file or (
        ROOT / "runs/under10m_screening/logs" / f"rung-{args.rung}.log"
    )
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--rung", str(args.rung)]
    if args.ids:
        command.extend(["--ids", *args.ids])
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(
        json.dumps(
            {"pid": process.pid, "rung": args.rung, "log": str(log_path)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
