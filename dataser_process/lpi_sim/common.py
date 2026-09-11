"""配置、稳定种子和不可覆盖的原子文件操作。"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed_for(seed, *parts):
    return int(digest([int(seed), *parts])[:16], 16)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(ROOT.rglob("*.py"))})


def atomic_bytes(path, data, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and immutable:
        if path.read_bytes() != data:
            raise ValueError(f"拒绝覆盖内容不一致的已有文件：{path}")
        return
    fd, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value, immutable=True):
    atomic_bytes(path, (canonical(value) + "\n").encode(), immutable)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPOSITORY / path
    cfg = yaml.safe_load(path.read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    expected = {"schema_version", "seed", "output", "snr_db", "splits", "signal", "stft", "annotation"}
    if not isinstance(cfg, dict) or set(cfg) != expected or cfg["schema_version"] != 1:
        raise ValueError("配置字段或 schema_version 不正确。")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed 必须为非负整数。")
    if not isinstance(cfg["output"], str) or not cfg["output"].strip():
        raise ValueError("output 必须是非空路径。")
    levels = cfg["snr_db"]
    if not isinstance(levels, list) or not levels or any(type(x) is not int for x in levels) or levels != sorted(set(levels)):
        raise ValueError("snr_db 必须为递增且无重复的整数列表。")
    if min(levels) < -100 or max(levels) > 100:
        raise ValueError("SNR 超出数值安全范围。")
    if set(cfg["splits"]) != {"train", "val", "test"}:
        raise ValueError("必须显式定义 train/val/test。")
    for counts in cfg["splits"].values():
        if set(counts) != {"scenes", "negatives"} or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError("子集数量必须是非负整数。")
        if counts["scenes"] % 30:
            raise ValueError("scenes 必须为 30 的倍数，才能保证目标数配额和五类平衡。")
    s, t, a = cfg["signal"], cfg["stft"], cfg["annotation"]
    if set(s) != {"fs_hz", "duration_s", "carrier_hz", "pulse_s", "margin_s", "bandwidth_hz", "bfsk_spacing_hz", "nlfm_curvature", "code_lengths", "frank_orders"}:
        raise ValueError("signal 字段不正确。")
    if set(t) != {"window", "hop", "nfft", "image_size", "db_range"} or set(a) != {"quantiles", "padding_px", "overlap_min", "max_attempts", "support_energy"}:
        raise ValueError("stft/annotation 字段不正确。")
    def pair(v):
        return isinstance(v, list) and len(v) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in v) and v[0] < v[1]
    for v in [s[k] for k in ("carrier_hz", "pulse_s", "bandwidth_hz", "bfsk_spacing_hz", "nlfm_curvature")] + [t["db_range"], a["quantiles"]]:
        if not pair(v):
            raise ValueError("区间必须是两个有限且递增的数。")
    for key in ("fs_hz", "duration_s", "margin_s"):
        if type(s[key]) not in (int, float) or not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"{key} 必须为有限正数。")
    n = round(s["fs_hz"] * s["duration_s"])
    if abs(n - s["fs_hz"] * s["duration_s"]) > 1e-6:
        raise ValueError("观测时长必须对应整数采样点。")
    if s["pulse_s"][0] <= 0 or s["pulse_s"][1] + 2*s["margin_s"] >= s["duration_s"]:
        raise ValueError("脉冲及边界余量必须能容纳在观测窗内。")
    if any(s[k][0] <= 0 for k in ("bandwidth_hz", "bfsk_spacing_hz", "nlfm_curvature")) or s["nlfm_curvature"][1] >= 1:
        raise ValueError("带宽/频差必须为正，NLFM 曲率必须在 (0,1) 内。")
    if max(abs(x) for x in s["carrier_hz"]) + max(s["bandwidth_hz"][1], s["bfsk_spacing_hz"][1])/2 >= s["fs_hz"]/2:
        raise ValueError("载频与调频带宽越过 Nyquist 边界。")
    for key, even in (("code_lengths", True), ("frank_orders", False)):
        values = s[key]
        if not isinstance(values, list) or len(values) != 3 or len(set(values)) != 3 or any(type(x) is not int or x < 4 or (even and x % 2) for x in values):
            raise ValueError(f"{key} 必须含三个互异整数，编码长度须为偶数。")
    if max(max(s["code_lengths"]), max(s["frank_orders"])**2) > math.floor(s["pulse_s"][0]*s["fs_hz"]):
        raise ValueError("每个码元至少需要一个采样点。")
    if any(type(t[k]) is not int or t[k] <= 0 for k in ("window", "hop", "nfft", "image_size")) or not 2 <= t["window"] <= t["nfft"] or t["window"] > n or t["hop"] > t["window"]:
        raise ValueError("STFT 尺寸不合法。")
    if t["nfft"] % 2 or not 0 <= a["quantiles"][0] < a["quantiles"][1] <= 1:
        raise ValueError("FFT 长度须为偶数；能量分位数须在 [0,1]。")
    if type(a["padding_px"]) is not int or a["padding_px"] < 0 or type(a["max_attempts"]) is not int or a["max_attempts"] < 1:
        raise ValueError("标注 padding/max_attempts 不合法。")
    if not 0 < a["overlap_min"] <= 1 or not 0 < a["support_energy"] < 1:
        raise ValueError("重叠阈值和支撑能量比例不合法。")
