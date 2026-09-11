#!/usr/bin/env python3
"""五类雷达时频数据生成入口；相对路径以仓库根目录解析。"""

import argparse
import json
import os
import sys

# 生成工作不需要 CUDA；限制底层库线程，避免多进程嵌套超额占用 CPU。
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

from lpi_sim.common import load_config
from lpi_sim.workflow import estimate, execute, preview_config, verify_existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dataser_process/configs/lpi5_v1.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "preview", "generate", "verify"):
        mode.add_argument("--"+name, action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers 必须为正整数。")
    if args.resume and not (args.generate or args.preview):
        parser.error("--resume 只能与 --generate 或 --preview 同时使用。")
    try:
        cfg = load_config(args.config)
        if args.preview:
            cfg = preview_config(cfg)
        if args.dry_run:
            result = estimate(cfg, args.workers)
        elif args.verify:
            result = verify_existing(cfg)
        else:
            resources = estimate(cfg, args.workers)
            if not args.resume and resources["free_bytes"] < resources["conservative_required_bytes"]:
                raise ValueError("可用磁盘空间低于未压缩 PNG 保守估算，请选择容量充足的输出位置。")
            result = execute(cfg, args.workers, args.resume, args.preview)
        # 完整分布写入质量报告，终端只显示摘要，避免输出数百行直方图。
        summary = {k: v for k, v in result.items() if k not in {"parameter_distributions", "discrete_parameter_counts"}}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
