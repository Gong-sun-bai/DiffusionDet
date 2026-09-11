"""生成事务、环境冻结和只读预检。"""

import copy
import fcntl
import importlib.metadata
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from . import VERSION
from .common import REPOSITORY, atomic_bytes, code_hash, digest, file_hash, read_json, seed_for, validate_config, write_json
from .generation import assemble, make_gallery, run_jobs
from .planning import quotas
from .validation import verify_dataset


def output_path(cfg):
    p = Path(cfg["output"])
    return p.resolve() if p.is_absolute() else (REPOSITORY/p).resolve()


def preview_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed_for(cfg["seed"], "independent-preview-v1")
    cfg["output"] = ".cache/lpi5_preview"
    cfg["splits"] = {"train": {"scenes": 30, "negatives": 10}, "val": {"scenes": 0, "negatives": 0}, "test": {"scenes": 0, "negatives": 0}}
    return cfg


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow", "PyYAML")}}


def estimate(cfg, workers):
    root = output_path(cfg)
    parent = root
    while not parent.exists():
        parent = parent.parent
    q = quotas(cfg)
    images = sum(v["images"] for v in q.values())
    raw = images*cfg["stft"]["image_size"]**2*3
    # PNG 最差接近未压缩，另预留元数据与临时产物。
    reserved = int(raw*1.10 + images*16384)
    return {"output": str(root), "quotas": q, "images": images, "workers": workers,
            "uncompressed_rgb_bytes": raw, "conservative_required_bytes": reserved,
            "free_bytes": shutil.disk_usage(parent).free,
            "note": "PNG 实际大小、耗时以独立 preview 的 benchmark.json 为准；不代表全量已生成。"}


def verify_existing(cfg):
    root = output_path(cfg)
    manifest = read_json(root/"dataset_manifest.json")
    if manifest["config_sha256"] != digest(cfg):
        raise ValueError("待校验数据集与配置不一致。")
    frozen = yaml.safe_load((root/"generation_config.yaml").read_text())
    if frozen != cfg:
        raise ValueError("冻结配置不一致。")
    return verify_dataset(cfg, root, manifest)


def execute(cfg, workers=2, resume=False, preview=False):
    validate_config(cfg)
    root = output_path(cfg)
    existed = root.exists()
    if existed and not root.is_dir():
        raise ValueError(f"输出路径不是目录：{root}")
    if existed and any(root.iterdir()) and not resume:
        raise ValueError(f"目标目录非空，拒绝覆盖：{root}；未完成生成需显式 --resume。")
    if resume and not (root/"dataset_manifest.json").is_file():
        raise ValueError("续生成必须存在有效 dataset_manifest.json。")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root/".generation.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("该目录已有生成进程，不能并发写入。") from error
        fingerprint = code_hash()
        env = environment()
        manifest_path = root/"dataset_manifest.json"
        if resume:
            manifest = read_json(manifest_path)
            if manifest["config_sha256"] != digest(cfg) or manifest["generator_sha256"] != fingerprint or manifest["environment"] != env or manifest["preview"] != preview:
                raise ValueError("配置、生成器代码、环境或预览模式改变，拒绝续生成。请使用新的输出目录。")
            if manifest["status"] == "completed":
                print("数据集已完成，执行完整校验，不重新生成。", flush=True)
                return verify_existing(cfg)
        else:
            # 获取锁后重新检查，防止检查空目录与上锁之间的竞争。
            if any(p.name != ".generation.lock" for p in root.iterdir()):
                raise ValueError("上锁后发现已有产物，拒绝覆盖。")
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True, capture_output=True)
            status = subprocess.run(["git", "status", "--porcelain"], cwd=REPOSITORY, text=True, capture_output=True)
            manifest = {"schema_version": 1, "dataset_version": "lpi5-v1", "generator_version": VERSION,
                        "status": "generating", "preview": preview, "config_sha256": digest(cfg),
                        "generator_sha256": fingerprint, "environment": env, "git_commit": git.stdout.strip(),
                        "git_dirty": bool(status.stdout), "quotas": quotas(cfg)}
            write_json(manifest_path, manifest)
        # 仅清理本生成器原子写入在进程被杀时留下的临时文件。
        for partial in root.rglob(".partial-*"):
            if partial.is_file():
                partial.unlink()
        atomic_bytes(root/"generation_config.yaml", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode())
        start = time.monotonic()
        run_jobs(cfg, root, workers, preview)
        assemble(cfg, root)
        if preview:
            make_gallery(cfg, root)
        report = verify_dataset(cfg, root)
        write_json(root/"quality_report"/"validation.json", report)
        if preview:
            files = list((root/"images").rglob("*.png"))
            seconds = time.monotonic()-start
            benchmark = {"elapsed_seconds_including_validation": seconds, "images": len(files),
                         "mean_png_bytes": sum(p.stat().st_size for p in files)/len(files), "workers": workers,
                         "note": "预览包含干净中间数组与画廊开销，不能直接当作全量生成性能。"}
            write_json(root/"quality_report"/"benchmark.json", benchmark, immutable=False)
        artifacts = {}
        for folder in ("annotations", "metadata", "quality_report"):
            for p in sorted((root/folder).rglob("*")):
                if p.is_file() and not p.name.startswith(".partial-"):
                    artifacts[str(p.relative_to(root))] = file_hash(p)
        for name in ("generation_config.yaml", "dataset.yaml"):
            artifacts[name] = file_hash(root/name)
        manifest.update(status="completed", artifacts=artifacts)
        write_json(manifest_path, manifest, immutable=False)
        print(f"数据集完成并通过校验：{root}", flush=True)
        return report
