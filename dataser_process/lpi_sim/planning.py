"""确定性的类别组合配额与分层随机流。"""

import itertools
from collections import Counter

import numpy as np

from .common import seed_for


def balanced_combinations(k, count):
    combos = list(itertools.combinations(range(5), k))
    q, r = divmod(count, len(combos))
    result = combos * q
    if r:
        # C(5,2)、C(5,3) 余下的五项取循环轨道，每类出现次数相等。
        orbit = sorted({tuple(sorted((i+j) % 5 for j in range(k))) for i in range(5)})
        if r != len(orbit):
            raise ValueError("类别组合配额无法严格平衡。")
        result += orbit
    return result


def plan_jobs(cfg):
    jobs = []
    next_image = 1
    for split in ("train", "val", "test"):
        count = cfg["splits"][split]["scenes"]
        combos = []
        for k in range(1, 6):
            combos += balanced_combinations(k, count // (3 if k == 1 else 6))
        rng = np.random.default_rng(seed_for(cfg["seed"], split, "schedule"))
        rng.shuffle(combos)
        multi_indices = [i for i, c in enumerate(combos) if len(c) > 1]
        forced = set(rng.choice(multi_indices, len(multi_indices)//2, replace=False).tolist())
        component_occurrences = Counter()
        for index, classes in enumerate(combos):
            variants = []
            for c in classes:
                variants.append(component_occurrences[c])
                component_occurrences[c] += 1
            jobs.append({"scene_id": f"{split}-s{index:06d}", "split": split, "classes": list(classes),
                         "variants": variants, "force_overlap": index in forced, "image_start": next_image})
            next_image += len(cfg["snr_db"])
        for index in range(cfg["splits"][split]["negatives"]):
            jobs.append({"scene_id": f"{split}-n{index:06d}", "split": split, "classes": [],
                         "variants": [], "force_overlap": False, "image_start": next_image})
            next_image += 1
    return jobs


def quotas(cfg):
    result = {}
    for split, n in cfg["splits"].items():
        boxes = n["scenes"] * 8 // 3 * len(cfg["snr_db"])
        result[split] = {"clean_scenes": n["scenes"], "positive_images": n["scenes"] * len(cfg["snr_db"]),
                         "negative_images": n["negatives"], "images": n["scenes"] * len(cfg["snr_db"]) + n["negatives"],
                         "boxes": boxes, "boxes_per_class": boxes//5}
    return result
