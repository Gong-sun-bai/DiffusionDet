"""全量只读校验：配额、跨子集隔离、标签、图片和可追溯性。"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from . import CLASSES
from .common import digest, file_hash, read_json, seed_for
from .generation import check_artifacts, shard_path
from .planning import plan_jobs, quotas
from .time_frequency import yolo_text


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def verify_dataset(cfg, root, manifest=None):
    root = Path(root)
    expected = quotas(cfg)
    image_ids, anno_ids, fingerprints, image_hashes = set(), set(), {}, {}
    metadata_images = iter(jsonl(root/"metadata"/"images.jsonl"))
    metadata_scenes = iter(jsonl(root/"metadata"/"scenes.jsonl"))
    counts = {s: Counter() for s in cfg["splits"]}
    class_counts = {s: Counter() for s in cfg["splits"]}
    snr_counts = {s: Counter() for s in cfg["splits"]}
    k_counts = {s: Counter() for s in cfg["splits"]}
    variants = {s: {c: Counter() for c in CLASSES} for s in cfg["splits"]}
    distributions = {s: {c: {"carrier_hz": [], "pulse_s": [], "arrival_s": []} for c in CLASSES} for s in cfg["splits"]}
    forced = Counter()
    expected_files = set()
    max_snr_error = 0.0
    observed_boxes = []
    render_black, render_white = [], []
    coco = {s: read_json(root/"annotations"/f"instances_{s}.json") for s in cfg["splits"]}
    coco_images = {s: iter(coco[s]["images"]) for s in coco}
    coco_annos = {s: iter(coco[s]["annotations"]) for s in coco}
    categories = [{"id": i+1, "name": c} for i, c in enumerate(CLASSES)]
    for s in coco:
        require(coco[s]["categories"] == categories, "COCO 类别映射不正确。")
    jobs = plan_jobs(cfg)
    for number, job in enumerate(jobs, 1):
        record = read_json(shard_path(root, job))
        require(record["job"] == job, "场景计划不一致。")
        require(check_artifacts(root, record), f"场景缺少产物：{job['scene_id']}")
        expected_files.update(record["files"])
        expected_scene = {k: v for k, v in record.items() if k not in {"images", "files"}}
        require(next(metadata_scenes, None) == expected_scene, "scenes.jsonl 与场景记录不一致。")
        s = job["split"]
        require(len(record["parameters"]) == len(job["classes"]) == len(record["boxes"]), "场景分量数量不正确。")
        require([p["class_id"]-1 for p in record["parameters"]] == job["classes"], "场景类别与计划不一致。")
        require(len(record["component_fingerprints"]) == len(record["parameters"]), "分量指纹数量错误。")
        require(0 <= record["attempt"] < cfg["annotation"]["max_attempts"], "场景重试次数非法。")
        if job["classes"]:
            k_counts[s][len(job["classes"])] += 1
        if job["force_overlap"]:
            forced[s] += 1
            require(record["max_box_overlap"] >= cfg["annotation"]["overlap_min"], "指定重叠场景未满足条件。")
        for slot, (p, fingerprint) in enumerate(zip(record["parameters"], record["component_fingerprints"])):
            calculated = digest({k: v for k, v in p.items() if k not in {"parameter_seed", "code_seed", "phase_rad"}})
            require(calculated == fingerprint, "分量指纹错误。")
            require(fingerprint not in fingerprints, "发现复用的分量（忽略全局相位与随机种子）。")
            fingerprints[fingerprint] = s
            c = p["class_name"]
            require(c == CLASSES[p["class_id"]-1], "分量类别名称错误。")
            require(p["parameter_seed"] == seed_for(cfg["seed"], job["scene_id"], record["attempt"], slot, "parameters") and
                    p["code_seed"] == seed_for(cfg["seed"], job["scene_id"], record["attempt"], slot, "code"), "参数随机流错误。")
            signal = cfg["signal"]
            fs = signal["fs_hz"]
            require(signal["carrier_hz"][0] <= p["fc_hz"] <= signal["carrier_hz"][1], "载频超出共同范围。")
            require(signal["pulse_s"][0]-1e-12 <= p["length"]/fs <= signal["pulse_s"][1]+1e-12, "脉宽超出共同范围。")
            require(p["start"] >= np.ceil(signal["margin_s"]*fs) and p["start"]+p["length"] <= round(signal["duration_s"]*fs)-np.ceil(signal["margin_s"]*fs), "脉冲越过观测窗余量。")
            distributions[s][c]["carrier_hz"].append(p["fc_hz"])
            distributions[s][c]["pulse_s"].append(p["length"]/fs)
            distributions[s][c]["arrival_s"].append(p["start"]/fs)
            if c in ("LFM", "NLFM"):
                variants[s][c][f"direction:{p['direction']}"] += 1
            if c == "NLFM":
                variants[s][c][f"curvature_sign:{int(np.sign(p['curvature']))}"] += 1
            if c in ("BPSK", "BFSK"):
                variants[s][c][f"chips:{len(p['code'])}"] += 1
            if c == "Frank":
                variants[s][c][f"order:{p['order']}"] += 1
        levels = cfg["snr_db"] if job["classes"] else [None]
        require(len(record["images"]) == len(levels), "场景 SNR 版本数量错误。")
        for offset, (row, snr) in enumerate(zip(record["images"], levels)):
            image_id = job["image_start"]+offset
            require(row["id"] == image_id and image_id not in image_ids, "图片 ID 错误或重复。")
            image_ids.add(image_id)
            require(row["scene_id"] == job["scene_id"] and row["split"] == s and row["snr_db"] == snr, "图片身份或 SNR 错误。")
            require(row["target_count"] == len(job["classes"]) and row["noise_type"] == "awgn", "目标数或噪声类型错误。")
            file_name = f"images/{s}/{image_id:09d}.png"
            require(row["file_name"] == file_name, "图片路径不符合规范。")
            require(next(metadata_images, None) == {k: v for k, v in row.items() if k != "annotations"}, "images.jsonl 不一致。")
            require(next(coco_images[s], None) == {k: row[k] for k in ("id", "file_name", "width", "height", "scene_id", "snr_db")}, "COCO 图片记录不一致。")
            require(row["width"] == row["height"] == cfg["stft"]["image_size"], "图片记录尺寸错误。")
            with Image.open(root/file_name) as im:
                require(im.format == "PNG" and im.mode == "RGB" and im.size == (row["width"], row["height"]), "图片编码或尺寸错误。")
                pixels = np.asarray(im)
                require(np.array_equal(pixels[:, :, 0], pixels[:, :, 1]) and np.array_equal(pixels[:, :, 0], pixels[:, :, 2]), "输入不是三通道相同的灰度图。")
            checksum = record["files"][file_name]
            require(checksum not in image_hashes, f"发现重复 PNG：{file_name}")
            image_hashes[checksum] = s
            counts[s]["images"] += 1
            counts[s]["positive_images" if snr is not None else "negative_images"] += 1
            if snr is not None:
                snr_counts[s][snr] += 1
            measurement = row["measurements"]
            require(measurement["noise_seed"] == seed_for(cfg["seed"], job["scene_id"], "noise", snr), "噪声随机流错误。")
            require(len(measurement["components"]) == len(job["classes"]), "实测 SNR 数量错误。")
            for m in measurement["components"]:
                err = abs(m["snr_db"]-snr)
                require(np.isfinite(err) and err < .01, "实测 SNR 超出误差要求。")
                max_snr_error = max(max_snr_error, err)
            require(len(row["annotations"]) == len(job["classes"]), "标注数量错误。")
            for slot, a in enumerate(row["annotations"]):
                require(next(coco_annos[s], None) == a, "COCO annotation 与场景记录不一致。")
                require(a["id"] == image_id*10+slot+1 and a["id"] not in anno_ids, "标注 ID 错误或重复。")
                anno_ids.add(a["id"])
                require(a["image_id"] == image_id and a["category_id"] == job["classes"][slot]+1, "标注所属图片/类别错误。")
                require(a["bbox"] == record["boxes"][slot], "同场景跨 SNR 的框发生变化。")
                x, y, w, h = a["bbox"]
                require(all(np.isfinite([x, y, w, h])) and x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= row["width"] and y+h <= row["height"], "框越界或退化。")
                require(a["area"] == w*h and a["iscrowd"] == 0, "COCO area/iscrowd 错误。")
                counts[s]["boxes"] += 1
                class_counts[s][a["category_id"]] += 1
                observed_boxes.append([w, h])
            label = root/"labels"/s/f"{image_id:09d}.txt"
            require(label.read_text() == yolo_text(row["annotations"], row["width"]), "YOLO 标签不是由 COCO 派生。")
            for line, a in zip(label.read_text().splitlines(), row["annotations"]):
                c, cx, cy, w, h = map(float, line.split())
                reconstructed = np.array([cx-w/2, cy-h/2, w, h])*row["width"]
                require(int(c) == a["category_id"]-1 and np.max(np.abs(reconstructed-a["bbox"])) <= .5, "YOLO 往返误差超限。")
            render_black.append(row["render_stats"]["black_fraction"])
            render_white.append(row["render_stats"]["white_fraction"])
            require(row["render_stats"]["black_fraction"] == float(np.mean(pixels[:, :, 0] == 0)) and
                    row["render_stats"]["white_fraction"] == float(np.mean(pixels[:, :, 0] == 255)), "图像饱和统计不一致。")
        if number % 500 == 0:
            print(f"校验场景 {number}/{len(jobs)}", flush=True)
    require(next(metadata_images, None) is None and next(metadata_scenes, None) is None, "元数据存在额外记录。")
    for s, q in expected.items():
        require(next(coco_images[s], None) is None and next(coco_annos[s], None) is None, "COCO 存在额外记录。")
        for key in ("images", "positive_images", "negative_images", "boxes"):
            require(counts[s][key] == q[key], f"{s} 的 {key} 不符合配额。")
        require(all(class_counts[s][i] == q["boxes_per_class"] for i in range(1, 6)), "五类实例未严格平衡。")
        require(all(snr_counts[s][v] == q["clean_scenes"] for v in cfg["snr_db"]), "SNR 数量不平衡。")
        require(all(k_counts[s][k] == q["clean_scenes"]//(3 if k == 1 else 6) for k in range(1, 6)), "目标数量配额错误。")
        require(forced[s] == q["clean_scenes"]//3, "指定重叠场景数量错误。")
    actual_files = {str(p.relative_to(root)) for folder in ("images", "labels") for p in (root/folder).rglob("*") if p.is_file()}
    require(actual_files == {p for p in expected_files if p.startswith(("images/", "labels/"))}, "图片或标签目录存在额外/缺少文件。")
    require({p.name for p in (root/"metadata"/"shards").glob("*.json")} == {j["scene_id"]+".json" for j in jobs}, "场景检查点存在额外/缺少文件。")
    expected_split_manifest = {"quotas": expected, "scene_ids": {s: [j["scene_id"] for j in jobs if j["split"] == s] for s in cfg["splits"]}}
    require(read_json(root/"metadata"/"split_manifest.json") == expected_split_manifest, "划分清单错误。")
    if manifest and "artifacts" in manifest:
        for rel, checksum in manifest["artifacts"].items():
            require((root/rel).is_file() and file_hash(root/rel) == checksum, f"清单哈希不一致：{rel}")
    dims = np.asarray(observed_boxes)
    distribution_summary = {s: {c: {key: ({"min": min(v), "max": max(v), "mean": float(np.mean(v)),
                                          "std": float(np.std(v)), "histogram_10_bins": np.histogram(v, bins=10)[0].tolist(),
                                          "histogram_edges": np.histogram(v, bins=10)[1].tolist()} if v else None)
                                      for key, v in entries.items()} for c, entries in classes.items()}
                            for s, classes in distributions.items()}
    return {"status": "passed", "quotas": expected, "snr_counts": {s: dict(v) for s, v in snr_counts.items()},
            "max_component_snr_error_db": max_snr_error, "unique_components": len(fingerprints),
            "unique_pngs": len(image_hashes), "discrete_parameter_counts": variants,
            "parameter_distributions": distribution_summary,
            "bbox_width_height_min": dims.min(axis=0).tolist() if dims.size else None,
            "bbox_width_height_median": np.median(dims, axis=0).tolist() if dims.size else None,
            "mean_black_fraction": float(np.mean(render_black)) if render_black else None,
            "mean_white_fraction": float(np.mean(render_white)) if render_white else None}
