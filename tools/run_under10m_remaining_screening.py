#!/usr/bin/env python3
"""Finish the under-10M local screening queue, without starting full training.

The queue is deliberately evidence-driven and resumable:

1. resume all seven candidates from their current checkpoints to 20k;
2. rank the exact 20k validation, keep the top three plus an optional AP75 leader;
3. resume only those candidates to 50k;
4. conditionally screen query-consistent distillation on the best 50k student;
5. write JSON/Markdown evidence and stop for human/Codex analysis.

This program never starts a 254264-iteration full experiment and never creates
the final full-training queue.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.experiment_run_utils import (  # noqa: E402
    action_for,
    atomic_write_json,
    load_experiment,
    read_json,
)
from tools.run_under10m_screening import CANDIDATES  # noqa: E402
from tools.set_screening_status import set_status  # noqa: E402
from tools.summarize_screening import (  # noqa: E402
    compare_rows,
    finite_number,
    summarize,
)


SCREENING_ROOT = ROOT / "runs/under10m_screening"
PROFILE_DIR = ROOT / "runs/screening_profiles"
SUMMARY_DIR = SCREENING_ROOT / "summaries"
STATE_PATH = SCREENING_ROOT / "remaining-screening-state.json"
LOCK_PATH = SCREENING_ROOT / ".remaining-screening.lock"
DEFAULT_LOG_PATH = SCREENING_ROOT / "logs/remaining-screening.log"
DISTILL_CONFIG = (
    ROOT
    / "configs/experiments/sar_ship/"
    / "sar-014-under10m-best-distilled-screen.yaml"
)
TEACHER_WEIGHTS = (
    ROOT
    / "runs/sar_ship/"
    / "sar-006__r18-fpn128-h128-bs25-it254264-fixed256-seed40244023/"
    / "model_best.pth"
)

DISTILL_RUNGS = (200, 20000, 50000)
MEMORY_LIMIT_MIB = 20 * 1024
STEP_TIME_RATIO_LIMIT = 1.8
GAIN_LIMIT = 1.0
AP_DROP_LIMIT = 0.3
FULL_MAX_ITER = 254264
FULL_EVALUATION_HOURS_PER_RUN = 4.0
FULL_DEADLINE_HOURS = 336.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_config(experiment_id: str) -> Path:
    return (ROOT / CANDIDATES[experiment_id]).resolve()


def candidate_configs(experiment_ids=None) -> list[Path]:
    selected = experiment_ids or sorted(CANDIDATES)
    return [candidate_config(experiment_id) for experiment_id in selected]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state() -> dict:
    return read_json(STATE_PATH) if STATE_PATH.is_file() else {}


def update_state(**changes) -> dict:
    state = load_state()
    state.update(changes)
    state["updated_at"] = utc_now()
    atomic_write_json(STATE_PATH, state)
    return state


def current_candidate_hashes() -> dict[str, str]:
    return {
        experiment_id: load_experiment(candidate_config(experiment_id)).config_hash
        for experiment_id in sorted(CANDIDATES)
    }


def guard_candidate_hashes(dry_run: bool) -> None:
    current = current_candidate_hashes()
    state = load_state()
    recorded = state.get("candidate_config_hashes")
    if recorded is not None and recorded != current:
        changed = sorted(
            experiment_id
            for experiment_id in set(recorded) | set(current)
            if recorded.get(experiment_id) != current.get(experiment_id)
        )
        raise RuntimeError(
            "candidate config changed after remaining screening started: "
            + ", ".join(changed)
        )
    if recorded is None and not dry_run:
        update_state(
            schema_version=1,
            started_at=state.get("started_at", utc_now()),
            candidate_config_hashes=current,
            orchestrator_sha256=sha256_file(Path(__file__).resolve()),
            stage="initialized",
        )


def gpu_preflight(min_free_gib: float) -> None:
    if not TEACHER_WEIGHTS.is_file():
        raise RuntimeError(f"teacher checkpoint does not exist: {TEACHER_WEIGHTS}")
    free_gib = shutil.disk_usage(ROOT).free / (1024**3)
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"insufficient free disk: {free_gib:.1f} GiB < {min_free_gib:.1f} GiB"
        )
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    probe = torch.ones(1, device="cuda")
    del probe
    torch.cuda.synchronize()
    print(f"[preflight] GPU={query.stdout.strip()} free_disk={free_gib:.1f}GiB")


def has_exact_rung(config: Path, rung: int) -> bool:
    return bool(summarize(config.resolve(), PROFILE_DIR, rung)["has_requested_rung"])


def run_no_distill_rung(rung: int, experiment_ids, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "tools/run_under10m_screening.py"),
        "--rung",
        str(rung),
    ]
    if experiment_ids:
        command.extend(["--ids", *experiment_ids])
    if dry_run:
        command.append("--dry-run")
    print(f"[queue] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_single_rung(config: Path, rung: int, dry_run: bool) -> None:
    if has_exact_rung(config, rung):
        print(f"[skip-rung] {config.name} already has exact iteration={rung}")
        return
    experiment = load_experiment(config)
    action = action_for(experiment)
    command = [
        sys.executable,
        str(ROOT / "train_net.py"),
        "--num-gpus",
        "1",
        "--config-file",
        str(config.resolve()),
        "--run-until-iter",
        str(rung),
    ]
    if action == "resume":
        command.append("--resume")
    elif action == "skip":
        if has_exact_rung(config, rung):
            return
        raise RuntimeError(f"completed experiment lacks exact rung={rung}: {config}")
    print(f"[{action}] {' '.join(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)
    if not has_exact_rung(config, rung):
        raise RuntimeError(f"training returned without exact rung={rung}: {config}")


def ranked_rows(configs: list[Path], rung: int) -> list[dict]:
    rows = [summarize(config.resolve(), PROFILE_DIR, rung) for config in configs]
    missing = [row["id"] for row in rows if not row["has_requested_rung"]]
    if missing:
        raise RuntimeError(
            f"missing exact iteration={rung} validation for: " + ", ".join(missing)
        )
    rows.sort(key=functools.cmp_to_key(compare_rows))
    return rows


def write_rung_summary(rows: list[dict], rung: int) -> tuple[Path, Path]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    json_path = SUMMARY_DIR / f"rung-{rung}.json"
    csv_path = SUMMARY_DIR / f"rung-{rung}.csv"
    encoded = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    temporary = json_path.with_name(f".{json_path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(json_path)
    csv_temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    csv_temporary.replace(csv_path)
    return json_path, csv_path


def select_promoted_ids(rows: list[dict]) -> list[str]:
    """Keep the ranked top three and an optional distinct AP75 leader."""
    promoted = [row["id"] for row in rows[:3]]
    ap75_rows = [row for row in rows if finite_number(row.get("AP75"))]
    if ap75_rows:
        ap75_leader = max(ap75_rows, key=lambda row: float(row["AP75"]))["id"]
        if ap75_leader not in promoted:
            promoted.append(ap75_leader)
    return promoted[:4]


def record_status(config: Path, status: str, reason: str) -> None:
    experiment = load_experiment(config)
    manifest_path = experiment.run_dir / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        history = manifest.get("screening_history") or []
        if any(
            decision.get("status") == status and decision.get("reason") == reason
            for decision in history
        ):
            return
    set_status(config, status, reason)


def classify_20k(rows: list[dict], promoted_ids: list[str]) -> None:
    rank_by_id = {row["id"]: index + 1 for index, row in enumerate(rows)}
    promoted_set = set(promoted_ids)
    for row in rows:
        experiment_id = row["id"]
        metrics = f"AP={row['AP']:.4f}、AP75={row['AP75']:.4f}、APs={row['APs']:.4f}"
        if experiment_id in promoted_set:
            reason = (
                f"20k 综合排名第 {rank_by_id[experiment_id]}，{metrics}；"
                "进入最多四个候选的 50k 筛选。"
            )
            record_status(candidate_config(experiment_id), "promoted", reason)
        else:
            reason = (
                f"20k 综合排名第 {rank_by_id[experiment_id]}，{metrics}；"
                "未进入前三且不是额外 AP75 领先候选。"
            )
            record_status(candidate_config(experiment_id), "screened_out", reason)


def classify_50k(rows: list[dict]) -> list[str]:
    provisional = [row["id"] for row in rows[:2]]
    for index, row in enumerate(rows, 1):
        metrics = f"AP={row['AP']:.4f}、AP75={row['AP75']:.4f}、APs={row['APs']:.4f}"
        if row["id"] in provisional:
            reason = (
                f"50k 综合排名第 {index}，{metrics}；列为无蒸馏 provisional finalist，"
                "等待蒸馏门槛与最终人工分析。"
            )
            record_status(candidate_config(row["id"]), "promoted", reason)
        else:
            reason = f"50k 综合排名第 {index}，{metrics}；未进入无蒸馏前二。"
            record_status(candidate_config(row["id"]), "screened_out", reason)
    return provisional


def render_distillation_config(student_row: dict, destination: Path = DISTILL_CONFIG) -> str:
    student_config = (ROOT / student_row["config"]).resolve()
    base = os.path.relpath(student_config, destination.parent)
    teacher = os.path.relpath(TEACHER_WEIGHTS, ROOT)
    return f'''_BASE_: "{base}"

EXPERIMENT:
  ID: "sar-014"
  NAME: "under10m-best-student-distilled-screen"
  DATASET: "sar_ship"
  DESCRIPTION: "由剩余筛选脚本基于 50k 最强无蒸馏学生 {student_row['id']} 生成；仅测试 sar-006 查询一致蒸馏，使用相同完整 schedule 并按 200/20k/50k 暂停。"
  OUTPUT_ROOT: "./runs"

MODEL:
  DiffusionDet:
    DISTILLATION:
      ENABLED: True
      TEACHER_WEIGHTS: "{teacher}"
      CONFIDENCE_THRESHOLD: 0.5
      TEMPERATURE: 2.0
      CLS_WEIGHT: 1.0
      L1_WEIGHT: 1.0
      GIOU_WEIGHT: 1.0
'''


def ensure_distillation_config(student_row: dict, dry_run: bool) -> Path:
    expected = render_distillation_config(student_row)
    if DISTILL_CONFIG.is_file():
        actual = DISTILL_CONFIG.read_text(encoding="utf-8")
        if actual != expected:
            raise RuntimeError(
                f"generated distillation config differs from the selected student: "
                f"{DISTILL_CONFIG}"
            )
    elif dry_run:
        print(f"[dry-run] would create {DISTILL_CONFIG.relative_to(ROOT)}")
    else:
        DISTILL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        temporary = DISTILL_CONFIG.with_name(f".{DISTILL_CONFIG.name}.tmp")
        temporary.write_text(expected, encoding="utf-8")
        temporary.replace(DISTILL_CONFIG)
    return DISTILL_CONFIG


def training_step_median(run_dir: Path, upper_iteration: int | None = None) -> float:
    metrics_path = run_dir / "metrics.json"
    times = []
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iteration = row.get("iteration")
            value = row.get("time")
            if (
                isinstance(iteration, int)
                and (upper_iteration is None or iteration < upper_iteration)
                and finite_number(value)
            ):
                times.append(float(value))
    if not times:
        raise RuntimeError(f"no finite training step time in {metrics_path}")
    times.sort()
    middle = len(times) // 2
    return (
        times[middle]
        if len(times) % 2
        else (times[middle - 1] + times[middle]) / 2.0
    )


def parse_peak_memory_mib(log_text: str) -> int | None:
    values = [int(value) for value in re.findall(r"max_mem:\s*(\d+)M", log_text)]
    return max(values) if values else None


def peak_memory_mib(run_dir: Path) -> int:
    log_path = run_dir / "log.txt"
    if not log_path.is_file():
        raise RuntimeError(f"training log does not exist: {log_path}")
    value = parse_peak_memory_mib(log_path.read_text(encoding="utf-8", errors="replace"))
    if value is None:
        raise RuntimeError(f"no max_mem evidence in {log_path}")
    return value


def is_resource_failure(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "cuda out of memory",
            "out of memory",
            "cublas_status_alloc_failed",
            "cuda error: memory allocation",
        )
    )


def manifest_error_text(config: Path) -> str:
    experiment = load_experiment(config)
    manifest_path = experiment.run_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        return ""
    error = read_json(manifest_path).get("error") or {}
    return f"{error.get('type', '')}: {error.get('message', '')}".strip()


def resource_gate(student_config: Path, distill_config: Path) -> dict:
    student = load_experiment(student_config)
    distilled = load_experiment(distill_config)
    student_step = training_step_median(student.run_dir, 200)
    distilled_step = training_step_median(distilled.run_dir, 200)
    peak = peak_memory_mib(distilled.run_dir)
    ratio = distilled_step / student_step
    return {
        "student_step_seconds_200": student_step,
        "distilled_step_seconds_200": distilled_step,
        "step_time_ratio": ratio,
        "step_time_ratio_limit": STEP_TIME_RATIO_LIMIT,
        "peak_memory_mib": peak,
        "memory_limit_mib": MEMORY_LIMIT_MIB,
        "passed": peak <= MEMORY_LIMIT_MIB and ratio <= STEP_TIME_RATIO_LIMIT,
    }


def eta_hours_for_three(rate_a: float, rate_b: float) -> float:
    training = (rate_a + rate_b + max(rate_a, rate_b)) * FULL_MAX_ITER / 3600.0
    return training + 3.0 * FULL_EVALUATION_HOURS_PER_RUN


def evaluate_distillation_gates(
    student_row: dict,
    distilled_row: dict,
    resource: dict,
    student_step_50k: float,
    distilled_step_50k: float,
) -> dict:
    ap_gain = float(distilled_row["AP"]) - float(student_row["AP"])
    ap75_gain = float(distilled_row["AP75"]) - float(student_row["AP75"])
    eta_hours = eta_hours_for_three(student_step_50k, distilled_step_50k)
    gain_passed = ap_gain >= GAIN_LIMIT or ap75_gain >= GAIN_LIMIT
    ap_floor_passed = ap_gain >= -AP_DROP_LIMIT
    eta_passed = eta_hours <= FULL_DEADLINE_HOURS
    passed = bool(resource["passed"] and gain_passed and ap_floor_passed and eta_passed)
    return {
        "resource_passed": bool(resource["passed"]),
        "AP_gain": ap_gain,
        "AP75_gain": ap75_gain,
        "gain_required": GAIN_LIMIT,
        "gain_passed": gain_passed,
        "AP_drop_limit": AP_DROP_LIMIT,
        "AP_floor_passed": ap_floor_passed,
        "estimated_three_full_runs_hours": eta_hours,
        "deadline_hours": FULL_DEADLINE_HOURS,
        "eta_passed": eta_passed,
        "eligible_to_replace_second_finalist": passed,
    }


def write_final_report(payload: dict) -> tuple[Path, Path]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    json_path = SUMMARY_DIR / "remaining-screening-result.json"
    markdown_path = SUMMARY_DIR / "remaining-screening-report.md"
    atomic_write_json(json_path, payload)

    rows = payload["no_distill_50k_ranking"]
    lines = [
        "# 10M 以下剩余本地筛选结果",
        "",
        f"> 生成时间：{payload['finished_at']}",
        "> 本报告只汇总 20k/50k 与条件蒸馏证据；未启动 254264 iter 正式训练。",
        "",
        "## 50k 无蒸馏排名",
        "",
        "| 排名 | ID | AP | AP75 | APs | step(s) | 参数量 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | `{row['id']}` | {row['AP']:.4f} | "
            f"{row['AP75']:.4f} | {row['APs']:.4f} | "
            f"{row['median_step_seconds']:.4f} | {row.get('trainable_parameters')} |"
        )
    lines.extend(
        [
            "",
            "## 条件蒸馏",
            "",
            f"- 状态：`{payload['distillation']['status']}`",
            f"- 学生：`{payload['distillation'].get('student_id')}`",
            f"- 是否具备替代第二 finalist 的门槛："
            f"`{payload['distillation'].get('eligible_to_replace_second_finalist', False)}`",
            "",
            "## 后续动作",
            "",
            "请通知 Codex 分析本报告与原始 metrics/log；之后再生成独立正式配置，由用户逐个启动。",
            "本脚本没有启动任何 254264 iter 正式实验。",
            "",
        ]
    )
    temporary = markdown_path.with_name(f".{markdown_path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(markdown_path)
    return json_path, markdown_path


def finish_without_distillation(
    rows_20k: list[dict],
    rows_50k: list[dict],
    promoted_20k: list[str],
    provisional: list[str],
    distillation: dict,
) -> dict:
    payload = {
        "schema_version": 1,
        "finished_at": utc_now(),
        "full_training_started": False,
        "promoted_at_20k": promoted_20k,
        "provisional_no_distill_finalists": provisional,
        "no_distill_20k_ranking": rows_20k,
        "no_distill_50k_ranking": rows_50k,
        "distillation": distillation,
        "next_action": "notify Codex to analyze evidence and generate standalone final configs",
    }
    json_path, markdown_path = write_final_report(payload)
    update_state(
        stage="complete_waiting_for_analysis",
        finished_at=payload["finished_at"],
        result_json=str(json_path.relative_to(ROOT)),
        result_markdown=str(markdown_path.relative_to(ROOT)),
        distillation=distillation,
    )
    print(f"[complete] {json_path.relative_to(ROOT)}")
    print(f"[complete] {markdown_path.relative_to(ROOT)}")
    print("[stop] no 254264-iteration full training was started")
    return payload


def run_pipeline(args) -> None:
    if not args.dry_run:
        gpu_preflight(args.min_free_gib)
    guard_candidate_hashes(args.dry_run)

    if not all(has_exact_rung(config, 20000) for config in candidate_configs()):
        run_no_distill_rung(20000, None, args.dry_run)
        if args.dry_run:
            print("[dry-run] stop: 20k evidence is needed before automatic promotion")
            return
    rows_20k = ranked_rows(candidate_configs(), 20000)
    write_rung_summary(rows_20k, 20000)
    promoted_20k = select_promoted_ids(rows_20k)
    print(f"[20k-promoted] {' '.join(promoted_20k)}")
    if not args.dry_run:
        classify_20k(rows_20k, promoted_20k)
        update_state(stage="20k_classified", promoted_at_20k=promoted_20k)

    promoted_configs = candidate_configs(promoted_20k)
    if not all(has_exact_rung(config, 50000) for config in promoted_configs):
        run_no_distill_rung(50000, promoted_20k, args.dry_run)
        if args.dry_run:
            print("[dry-run] stop: 50k evidence is needed before distillation selection")
            return
    rows_50k = ranked_rows(promoted_configs, 50000)
    write_rung_summary(rows_50k, 50000)
    provisional = [row["id"] for row in rows_50k[:2]]
    if not args.dry_run:
        provisional = classify_50k(rows_50k)
        update_state(
            stage="50k_classified",
            provisional_no_distill_finalists=provisional,
        )
    print(f"[50k-provisional] {' '.join(provisional)}")

    strongest = rows_50k[0]
    if args.skip_distillation:
        if args.dry_run:
            print("[dry-run] distillation would be skipped by request")
            return
        finish_without_distillation(
            rows_20k,
            rows_50k,
            promoted_20k,
            provisional,
            {
                "status": "skipped_by_request",
                "student_id": strongest["id"],
                "eligible_to_replace_second_finalist": False,
            },
        )
        return

    distill_config = ensure_distillation_config(strongest, args.dry_run)
    if args.dry_run:
        print(
            f"[dry-run] would run {distill_config.relative_to(ROOT)} at rungs "
            + "/".join(str(value) for value in DISTILL_RUNGS)
        )
        return

    state = load_state()
    prior_distill = state.get("distillation") or {}
    if prior_distill.get("status") == "rejected_resource":
        finish_without_distillation(
            rows_20k, rows_50k, promoted_20k, provisional, prior_distill
        )
        return

    try:
        run_single_rung(distill_config, 200, False)
    except subprocess.CalledProcessError:
        error_text = manifest_error_text(distill_config)
        if not is_resource_failure(error_text):
            raise
        distilled = load_experiment(distill_config)
        distillation = {
            "status": "rejected_resource",
            "student_id": strongest["id"],
            "config": str(distill_config.relative_to(ROOT)),
            "reason": error_text,
            "eligible_to_replace_second_finalist": False,
        }
        update_state(stage="distillation_rejected_resource", distillation=distillation)
        finish_without_distillation(
            rows_20k, rows_50k, promoted_20k, provisional, distillation
        )
        print(f"[distillation-resource-rejected] {distilled.run_dir}")
        return

    resource = resource_gate(candidate_config(strongest["id"]), distill_config)
    if not resource["passed"]:
        reason = (
            f"蒸馏 200 iter 资源门槛未通过：peak={resource['peak_memory_mib']}MiB/"
            f"{MEMORY_LIMIT_MIB}MiB，step ratio={resource['step_time_ratio']:.3f}/"
            f"{STEP_TIME_RATIO_LIMIT:.1f}。"
        )
        record_status(distill_config, "screened_out", reason)
        distillation = {
            "status": "rejected_resource",
            "student_id": strongest["id"],
            "config": str(distill_config.relative_to(ROOT)),
            "resource": resource,
            "reason": reason,
            "eligible_to_replace_second_finalist": False,
        }
        update_state(stage="distillation_rejected_resource", distillation=distillation)
        finish_without_distillation(
            rows_20k, rows_50k, promoted_20k, provisional, distillation
        )
        return

    record_status(
        distill_config,
        "promoted",
        f"蒸馏 200 iter 资源门槛通过：peak={resource['peak_memory_mib']}MiB，"
        f"step ratio={resource['step_time_ratio']:.3f}。",
    )
    update_state(stage="distillation_200_passed", distillation_resource=resource)
    for rung in (20000, 50000):
        run_single_rung(distill_config, rung, False)
        record_status(
            distill_config,
            "promoted",
            f"蒸馏候选已完成 exact {rung} iter 验证，继续执行预定条件筛选。",
        )
        update_state(stage=f"distillation_{rung}_completed")

    distilled_row = ranked_rows([distill_config], 50000)[0]
    distilled_experiment = load_experiment(distill_config)
    student_experiment = load_experiment(candidate_config(strongest["id"]))
    gates = evaluate_distillation_gates(
        strongest,
        distilled_row,
        resource,
        training_step_median(student_experiment.run_dir, 50000),
        training_step_median(distilled_experiment.run_dir, 50000),
    )
    eligible = gates["eligible_to_replace_second_finalist"]
    reason = (
        f"蒸馏 50k：AP gain={gates['AP_gain']:.4f}，"
        f"AP75 gain={gates['AP75_gain']:.4f}，"
        f"三次正式实验 ETA={gates['estimated_three_full_runs_hours']:.2f}h；"
        f"eligible={eligible}。"
    )
    record_status(distill_config, "promoted" if eligible else "screened_out", reason)
    distillation = {
        "status": "eligible" if eligible else "rejected_metrics_or_eta",
        "student_id": strongest["id"],
        "config": str(distill_config.relative_to(ROOT)),
        "row_50k": distilled_row,
        "resource": resource,
        **gates,
        "reason": reason,
    }
    finish_without_distillation(
        rows_20k, rows_50k, promoted_20k, provisional, distillation
    )


def detached_command(args) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    if args.skip_distillation:
        command.append("--skip-distillation")
    command.extend(["--min-free-gib", str(args.min_free_gib)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run independently and append all output to the screening log",
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--min-free-gib", type=float, default=15.0)
    parser.add_argument(
        "--skip-distillation",
        action="store_true",
        help="finish after no-distillation 50k (not the default plan)",
    )
    args = parser.parse_args()
    if args.detach:
        if args.dry_run:
            raise ValueError("--detach and --dry-run cannot be combined")
        log_path = args.log_file
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                detached_command(args),
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"pid": process.pid, "log": str(log_path)}, ensure_ascii=False))
        return

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another remaining-screening queue is already running") from error
        run_pipeline(args)


if __name__ == "__main__":
    main()
