"""复数雷达波形。所有频率均以 Hz、时间均以秒表示。"""

import numpy as np

from . import CLASSES
from .common import seed_for


def parameters(cfg, job, attempt):
    s = cfg["signal"]
    fs = s["fs_hz"]
    n = round(fs*s["duration_s"])
    margin = int(np.ceil(fs*s["margin_s"]))
    result = []
    for slot, (c, variant) in enumerate(zip(job["classes"], job["variants"])):
        parameter_seed = seed_for(cfg["seed"], job["scene_id"], attempt, slot, "parameters")
        code_seed = seed_for(cfg["seed"], job["scene_id"], attempt, slot, "code")
        rng = np.random.default_rng(parameter_seed)
        length = int(rng.integers(int(np.ceil(fs*s["pulse_s"][0])), int(np.floor(fs*s["pulse_s"][1]))+1))
        p = {"class_id": c+1, "class_name": CLASSES[c], "start": int(rng.integers(margin, n-margin-length+1)),
             "length": length, "fc_hz": float(rng.uniform(*s["carrier_hz"])), "phase_rad": float(rng.uniform(0, 2*np.pi)),
             "parameter_seed": parameter_seed, "code_seed": code_seed}
        # 离散因素按全子集出现序号分配，确保方向、编码长度、Frank 阶数均衡。
        if c in (0, 1):
            p.update(bandwidth_hz=float(rng.uniform(*s["bandwidth_hz"])), direction=(-1 if variant % 2 else 1))
            if c == 1:
                p["curvature"] = float(rng.uniform(*s["nlfm_curvature"])) * (-1 if (variant//2) % 2 else 1)
        if c in (2, 3):
            chips = s["code_lengths"][variant % 3]
            crng = np.random.default_rng(code_seed)
            code = np.tile([0, 1], chips//2)
            for _ in range(1000):
                crng.shuffle(code)
                changes = int(np.count_nonzero(np.diff(code)))
                if chips/4 <= changes <= 3*chips/4:
                    break
            else:
                raise ValueError("无法生成满足跳变约束的均衡码。")
            p["code"] = code.tolist()
            if c == 3:
                p["spacing_hz"] = float(rng.uniform(*s["bfsk_spacing_hz"]))
        if c == 4:
            p["order"] = s["frank_orders"][variant % 3]
        result.append(p)
    return result


def chip_indices(length, chips):
    edges = np.floor(np.arange(chips+1)*length/chips).astype(int)
    return np.repeat(np.arange(chips), np.diff(edges))


def waveform(p, cfg):
    fs = cfg["signal"]["fs_hz"]
    n = round(fs*cfg["signal"]["duration_s"])
    length = p["length"]
    t = np.arange(length, dtype=np.float64)/fs
    duration = length/fs
    u = t/duration
    cycles = p["fc_hz"]*t
    c = p["class_name"]
    if c in ("LFM", "NLFM"):
        a = p.get("curvature", 0)
        cycles += p["direction"]*p["bandwidth_hz"]*duration*(u*u/2 + a*(u*u/2-u*u*u/3)-u/2)
    elif c == "BPSK":
        bits = np.asarray(p["code"])[chip_indices(length, len(p["code"]))]
        cycles += bits/2
    elif c == "BFSK":
        bits = np.asarray(p["code"])[chip_indices(length, len(p["code"]))]
        freq = p["fc_hz"] + (bits-0.5)*p["spacing_hz"]
        cycles = np.concatenate(([0.0], np.cumsum(freq[:-1]/fs)))
    elif c == "Frank":
        m = p["order"]
        phases = (np.outer(np.arange(m), np.arange(m)) % m).reshape(-1)/m
        cycles += phases[chip_indices(length, m*m)]
    else:
        raise ValueError(f"未知类别：{c}")
    pulse = np.exp(1j*(2*np.pi*cycles+p["phase_rad"]))
    pulse /= np.sqrt(np.mean(np.abs(pulse)**2))
    result = np.zeros(n, dtype=np.complex128)
    result[p["start"]:p["start"]+length] = pulse
    return result


def noisy_scene(waves, params, cfg, noise_seed, snr):
    n = round(cfg["signal"]["fs_hz"]*cfg["signal"]["duration_s"])
    rng = np.random.default_rng(noise_seed)
    noise = (rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2)
    clean = np.zeros(n, dtype=np.complex128)
    active = np.zeros(n, dtype=bool)
    measurements = []
    for wave, p in zip(waves, params):
        region = slice(p["start"], p["start"]+p["length"])
        pn = float(np.mean(np.abs(noise[region])**2))
        ps = float(np.mean(np.abs(wave[region])**2))
        gain = float(np.sqrt(10**(snr/10)*pn/ps))
        component = gain*wave
        measured = float(10*np.log10(np.mean(np.abs(component[region])**2)/pn))
        measurements.append({"gain": gain, "noise_power_active": pn, "snr_db": measured})
        clean += component
        active[region] = True
    total = {}
    if waves:
        for name, region in (("full", slice(None)), ("active_union", active)):
            total[name] = float(10*np.log10(np.mean(np.abs(clean[region])**2)/np.mean(np.abs(noise[region])**2)))
    return clean+noise, {"noise_seed": noise_seed, "noise_power_full": float(np.mean(np.abs(noise)**2)),
                         "components": measurements, "mixture_snr_db": total}
