#!/usr/bin/env python3
"""Summarize and rank under-10M screening runs from metrics.json files."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.validate_experiment_configs import resolved_config
from experiment_manager import metadata_from_cfg, resolve_run_dir


def finite_number(value):
    return isinstance(value, (int, float)) and value == value and abs(value) != float("inf")


def metric_rows(path: Path):
    rows = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if finite_number(row.get("bbox/AP")):
            rows.append(row)
    return rows


def select_metric_row(rows, rung: int | None):
    """Select the exact rung result, or the best result for legacy summaries."""
    if rung is not None:
        matching = [row for row in rows if row.get("iteration") == rung]
        return matching[-1] if matching else {}
    return (
        max(
            rows,
            key=lambda row: tuple(
                float(row.get(key, float("-inf")))
                for key in ("bbox/AP", "bbox/AP75", "bbox/APs")
            ),
        )
        if rows
        else {}
    )


def summarize(config_path: Path, profile_dir: Path | None, rung: int | None = None):
    config = resolved_config(config_path)
    metadata = metadata_from_cfg(config)
    run_dir = resolve_run_dir(ROOT, metadata)
    rows = metric_rows(run_dir / "metrics.json")
    best = select_metric_row(rows, rung)
    all_rows = []
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                all_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    times = [float(row["time"]) for row in all_rows if finite_number(row.get("time"))]
    profile = {}
    if profile_dir:
        profile_path = profile_dir / f"{metadata.experiment_id}.json"
        if profile_path.is_file():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
    return {
        "id": metadata.experiment_id,
        "name": metadata.name,
        "config": str(config_path.relative_to(ROOT)),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "requested_rung": rung,
        "has_requested_rung": rung is None or bool(best),
        "best_iter": best.get("iteration"),
        "AP": best.get("bbox/AP"),
        "AP50": best.get("bbox/AP50"),
        "AP75": best.get("bbox/AP75"),
        "APs": best.get("bbox/APs"),
        "APm": best.get("bbox/APm"),
        "APl": best.get("bbox/APl"),
        "median_step_seconds": statistics.median(times) if times else None,
        "trainable_parameters": profile.get("trainable_parameters"),
        "latency_seconds_median": profile.get("latency_seconds_median"),
        "inference_peak_memory_mib": profile.get("peak_memory_mib"),
        "train_step_seconds": (profile.get("train_step") or {}).get("seconds"),
        "train_peak_memory_mib": (profile.get("train_step") or {}).get(
            "peak_memory_mib"
        ),
        "below_9_5m": profile.get("below_9_5m"),
    }


def compare_rows(left, right):
    left_ap = float(left["AP"]) if finite_number(left.get("AP")) else float("-inf")
    right_ap = float(right["AP"]) if finite_number(right.get("AP")) else float("-inf")
    if abs(left_ap - right_ap) > 0.5:
        return -1 if left_ap > right_ap else 1
    for key in ("AP75", "APs"):
        left_value = (
            float(left[key]) if finite_number(left.get(key)) else float("-inf")
        )
        right_value = (
            float(right[key]) if finite_number(right.get(key)) else float("-inf")
        )
        if left_value != right_value:
            return -1 if left_value > right_value else 1
    left_latency = float(left.get("latency_seconds_median") or float("inf"))
    right_latency = float(right.get("latency_seconds_median") or float("inf"))
    if left_latency == right_latency:
        return 0
    return -1 if left_latency < right_latency else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--rung",
        type=int,
        choices=(200, 5000, 20000, 50000),
        help="rank the exact rung instead of the best validation seen so far",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    configs = args.configs or sorted(
        (ROOT / "configs/experiments/sar_ship").glob("sar-0*-under10m-*.yaml")
    )
    rows = [summarize(path.resolve(), args.profile_dir, args.rung) for path in configs]
    if args.rung is not None:
        missing = [row["id"] for row in rows if not row["has_requested_rung"]]
        if missing:
            raise RuntimeError(
                f"missing exact iteration={args.rung} validation for: "
                + ", ".join(missing)
            )
    rows.sort(key=functools.cmp_to_key(compare_rows))
    encoded = json.dumps(rows, ensure_ascii=False, indent=2)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
