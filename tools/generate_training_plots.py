#!/usr/bin/env python3
"""Generate the canonical ``results.png`` for an existing experiment run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusiondet.training_visualization import (
    TrainingVisualizationError,
    generate_results_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取结果目录中的 metrics.json 并生成 YOLO 风格 results.png。",
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="单个 DiffusionDet 结果目录，例如 runs/sar_ship/sar-001__...。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = generate_results_plot(args.result_dir.expanduser().resolve())
    except TrainingVisualizationError as error:
        print(f"生成训练结果图失败：{error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"生成训练结果图失败：{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        "训练结果图生成完成："
        f"训练记录={summary.training_records}，"
        f"验证记录={summary.validation_records}，"
        f"最新迭代={summary.latest_iteration}，"
        f"输出={summary.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
