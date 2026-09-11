"""场景生成、原子检查点及数据集导出。"""

import io
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from . import CLASSES
from .common import atomic_bytes, digest, file_hash, read_json, seed_for, write_json
from .planning import plan_jobs, quotas
from .signals import noisy_scene, parameters, waveform
from .time_frequency import Spectrogram, box_overlap, energy_bbox, png_bytes, support_overlap, yolo_text


def shard_path(root, job):
    return Path(root)/"metadata"/"shards"/(job["scene_id"]+".json")


def check_artifacts(root, record):
    for rel, checksum in record["files"].items():
        p = Path(root)/rel
        if not p.is_file():
            return False
        if file_hash(p) != checksum:
            raise ValueError(f"已有产物哈希不一致，拒绝覆盖：{p}")
    return True


def generate_job(args):
    cfg, root, job, preview = args
    root = Path(root)
    shard = shard_path(root, job)
    if shard.exists():
        record = read_json(shard)
        if record["job"] != job:
            raise ValueError(f"场景计划发生变化：{shard}")
        if check_artifacts(root, record):
            return job["scene_id"]
    tf = Spectrogram(cfg)
    params, waves, powers, boxes = [], [], [], []
    attempt = 0
    for attempt in range(cfg["annotation"]["max_attempts"]):
        params = parameters(cfg, job, attempt)
        if job["force_overlap"]:
            # 条件场景对随机一对分量共享载频和 TOA；不按类别挑选锚点。
            # 其余参数仍独立采样，随后用真实干净框验证重叠。
            rng = np.random.default_rng(seed_for(cfg["seed"], job["scene_id"], attempt, "placement"))
            i, j = rng.choice(len(params), 2, replace=False)
            s = cfg["signal"]
            margin = int(np.ceil(s["margin_s"]*s["fs_hz"]))
            start = int(rng.integers(margin, tf.n-margin-max(params[i]["length"], params[j]["length"])+1))
            fc = float(rng.uniform(*s["carrier_hz"]))
            for k in (i, j):
                params[k]["start"], params[k]["fc_hz"] = start, fc
        waves = [waveform(p, cfg) for p in params]
        powers = [tf.power(w) for w in waves]
        boxes = [energy_bbox(p, cfg) for p in powers]
        max_overlap = max((box_overlap(a, b) for a, b in itertools.combinations(boxes, 2)), default=0)
        if not job["force_overlap"] or max_overlap >= cfg["annotation"]["overlap_min"]:
            break
    else:
        raise ValueError(f"{job['scene_id']} 在规定次数内无法满足重叠约束。")
    record = {"job": job, "attempt": attempt, "parameters": params, "boxes": boxes,
              "max_box_overlap": max_overlap,
              "energy_support_overlap": support_overlap(powers, cfg["annotation"]["support_energy"]),
              "component_fingerprints": [digest({k: v for k, v in p.items() if k not in {"parameter_seed", "code_seed", "phase_rad"}}) for p in params],
              "images": [], "files": {}}
    def emit(rel, data):
        atomic_bytes(root/rel, data)
        record["files"][rel] = file_hash(root/rel)
    levels = cfg["snr_db"] if params else [None]
    first_received = None
    for offset, snr in enumerate(levels):
        image_id = job["image_start"]+offset
        stem = f"{image_id:09d}"
        filename = f"images/{job['split']}/{stem}.png"
        noise_seed = seed_for(cfg["seed"], job["scene_id"], "noise", snr)
        received, measurements = noisy_scene(waves, params, cfg, noise_seed, snr)
        if first_received is None:
            first_received = received
        power = tf.power(received, agc=True)
        rgb, stats = tf.render(power)
        annotations = [{"id": image_id*10+i+1, "image_id": image_id, "category_id": p["class_id"],
                        "bbox": box, "area": box[2]*box[3], "iscrowd": 0}
                       for i, (p, box) in enumerate(zip(params, boxes))]
        emit(filename, png_bytes(rgb))
        emit(f"labels/{job['split']}/{stem}.txt", yolo_text(annotations, tf.size).encode())
        record["images"].append({"id": image_id, "file_name": filename, "width": tf.size, "height": tf.size,
                                  "scene_id": job["scene_id"], "split": job["split"], "snr_db": snr,
                                  "noise_type": "awgn", "target_count": len(params), "measurements": measurements,
                                  "render_stats": stats, "annotations": annotations})
    if preview:
        # 预览中保留干净 IQ、第一档接收 IQ、干净功率网格；正式集不存中间数组。
        arrays = {"clean_iq": np.asarray(waves, dtype=np.complex128), "received_first": first_received,
                  "clean_power": np.asarray(powers, dtype=np.float32)}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        emit(f"quality_report/intermediates/{job['scene_id']}.npz", buffer.getvalue())
    write_json(shard, record)
    return job["scene_id"]


def run_jobs(cfg, root, workers, preview=False):
    jobs = plan_jobs(cfg)
    start = time.monotonic()
    args = ((cfg, str(root), j, preview) for j in jobs)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(generate_job, args, chunksize=1) if pool else map(generate_job, args)
        for i, _ in enumerate(results, 1):
            if i == 1 or i % 25 == 0 or i == len(jobs):
                print(f"场景进度 {i}/{len(jobs)}，耗时 {time.monotonic()-start:.1f}s", flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)


def assemble(cfg, root):
    root = Path(root)
    categories = [{"id": i+1, "name": c} for i, c in enumerate(CLASSES)]
    coco = {s: {"info": {"description": "lpi5-v1"}, "licenses": [], "categories": categories,
                "images": [], "annotations": []} for s in cfg["splits"]}
    scenes, images = [], []
    for job in plan_jobs(cfg):
        record = read_json(shard_path(root, job))
        scenes.append({k: v for k, v in record.items() if k not in {"images", "files"}})
        for row in record["images"]:
            target = coco[row["split"]]
            target["images"].append({k: row[k] for k in ("id", "file_name", "width", "height", "scene_id", "snr_db")})
            target["annotations"].extend(row["annotations"])
            images.append({k: v for k, v in row.items() if k != "annotations"})
    from .common import canonical
    for split, value in coco.items():
        write_json(root/"annotations"/f"instances_{split}.json", value)
    for name, rows in (("scenes", scenes), ("images", images)):
        atomic_bytes(root/"metadata"/f"{name}.jsonl", "".join(canonical(r)+"\n" for r in rows).encode())
    write_json(root/"metadata"/"split_manifest.json", {"quotas": quotas(cfg), "scene_ids": {
        s: [j["scene_id"] for j in plan_jobs(cfg) if j["split"] == s] for s in cfg["splits"]}})
    dataset = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
               "names": dict(enumerate(CLASSES))}
    atomic_bytes(root/"dataset.yaml", yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False).encode())
    return images


def make_gallery(cfg, root):
    root = Path(root)
    selected = {}
    for job in plan_jobs(cfg):
        label = (CLASSES[job["classes"][0]] if len(job["classes"]) == 1 else
                 f"K={len(job['classes'])}" if job["classes"] else "Noise")
        selected.setdefault(label, read_json(shard_path(root, job)))
    cell, head = 160, 22
    columns = len(cfg["snr_db"])
    canvas = Image.new("RGB", (cell*columns, (cell+head)*len(selected)), "white")
    inputs = canvas.copy()
    clean_canvas = Image.new("RGB", (cell*5, cell+head), "white")
    draw = ImageDraw.Draw(canvas)
    input_draw = ImageDraw.Draw(inputs)
    clean_draw = ImageDraw.Draw(clean_canvas)
    colors = ("red", "lime", "cyan", "yellow", "magenta")
    order = [c for c in (*CLASSES, "K=2", "K=3", "K=4", "K=5", "Noise") if c in selected]
    tf = Spectrogram(cfg)
    for r, label in enumerate(order):
        record = selected[label]
        if label in CLASSES:
            with np.load(root/"quality_report"/"intermediates"/(record["job"]["scene_id"]+".npz")) as arrays:
                rgb, _ = tf.render(tf.power(arrays["clean_iq"][0], agc=True))
            tile = Image.fromarray(rgb)
            tile.thumbnail((cell, cell))
            clean_canvas.paste(tile, (CLASSES.index(label)*cell, head))
            clean_draw.text((CLASSES.index(label)*cell+2, 3), label+" clean", fill="black")
        for c, row in enumerate(record["images"]):
            with Image.open(root/row["file_name"]) as src:
                picture = src.copy()
            raw = picture.copy()
            raw.thumbnail((cell, cell))
            inputs.paste(raw, (c*cell, r*(cell+head)+head))
            input_draw.text((c*cell+2, r*(cell+head)+3), f"{label} / {row['snr_db']} dB", fill="black")
            pd = ImageDraw.Draw(picture)
            for a in row["annotations"]:
                x, y, w, h = a["bbox"]
                pd.rectangle((x, y, x+w-1, y+h-1), outline=colors[a["category_id"]-1], width=3)
                pd.text((x+2, y+2), CLASSES[a["category_id"]-1], fill=colors[a["category_id"]-1])
            picture.thumbnail((cell, cell))
            canvas.paste(picture, (c*cell, r*(cell+head)+head))
            draw.text((c*cell+2, r*(cell+head)+3), f"{label} / {row['snr_db']} dB", fill="black")
    for name, picture in (("preview_grid", canvas), ("preview_inputs", inputs), ("preview_clean", clean_canvas)):
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        atomic_bytes(root/"quality_report"/(name+".png"), buffer.getvalue())
