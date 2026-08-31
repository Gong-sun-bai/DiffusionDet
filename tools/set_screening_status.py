#!/usr/bin/env python3
"""Record a promoted or screened-out decision for a paused screening run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.experiment_run_utils import (
    atomic_write_json,
    load_experiment,
    read_json,
)


def set_status(config: Path, status: str, reason: str):
    experiment = load_experiment(config)
    manifest_path = experiment.run_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest does not exist: {manifest_path}")
    manifest = read_json(manifest_path)
    recorded_hash = manifest.get("execution", {}).get("config_sha256")
    if recorded_hash != experiment.config_hash:
        raise RuntimeError(
            f"config hash mismatch: recorded={recorded_hash!r}, "
            f"current={experiment.config_hash!r}"
        )
    current = manifest.get("status")
    if current not in {"paused", "promoted", "screened_out"}:
        raise RuntimeError(
            f"only paused screening runs may be classified, got {current!r}"
        )
    decision = {
        "status": status,
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    history = list(manifest.get("screening_history", ()))
    history.append(decision)
    manifest["screening_history"] = history
    manifest["status"] = status
    atomic_write_json(manifest_path, manifest)
    return experiment.run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--status", choices=("promoted", "screened_out"), required=True
    )
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    run_dir = set_status(args.config.resolve(), args.status, args.reason)
    print(json.dumps({"run_dir": str(run_dir), "status": args.status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
