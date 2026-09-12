#!/usr/bin/env python3
"""将本项目 SAR COCO 标注安全转换为 Ultralytics YOLO 检测数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train2017", "val2017")


class DatasetPreparationError(RuntimeError):
    """表示源数据或已有生成目录不符合预期。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(source_root: Path, split: str) -> dict:
    annotation_path = source_root / "annotations" / f"instances_{split}.json"
    image_dir = source_root / split
    if not annotation_path.is_file():
        raise DatasetPreparationError(f"标注文件不存在：{annotation_path}")
    if not image_dir.is_dir():
        raise DatasetPreparationError(f"图片目录不存在：{image_dir}")

    with annotation_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise DatasetPreparationError(f"COCO 标注顶层必须是映射：{annotation_path}")

    categories = data.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise DatasetPreparationError(f"SAR 数据必须且只能包含一个类别：{annotation_path}")
    category = categories[0]
    if category.get("name") != "ship" or not isinstance(category.get("id"), int):
        raise DatasetPreparationError(
            f"SAR 唯一类别必须命名为 ship 且使用整数 category_id：{annotation_path}"
        )
    category_id = category["id"]

    images = data.get("images")
    annotations = data.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise DatasetPreparationError(f"COCO images/annotations 必须是列表：{annotation_path}")

    image_by_id = {}
    stems = set()
    for image in images:
        image_id = image.get("id")
        file_name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if not isinstance(image_id, int):
            raise DatasetPreparationError(f"图片 ID 必须为整数：{image_id!r}")
        if image_id in image_by_id:
            raise DatasetPreparationError(f"图片 ID 重复：{image_id!r}")
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise DatasetPreparationError(f"file_name 必须是不含目录的相对文件名：{file_name!r}")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise DatasetPreparationError(f"图片尺寸无效：{file_name}")
        if Path(file_name).stem in stems:
            raise DatasetPreparationError(f"图片文件主名重复，无法唯一映射标签：{file_name}")
        if not (image_dir / file_name).is_file():
            raise DatasetPreparationError(f"JSON 引用的图片不存在：{image_dir / file_name}")
        stems.add(Path(file_name).stem)
        image_by_id[image_id] = image

    annotations_by_image = defaultdict(list)
    annotation_ids = set()
    for annotation in annotations:
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        if not isinstance(annotation_id, int) or not isinstance(image_id, int):
            raise DatasetPreparationError(
                f"标注 ID 和 image_id 必须为整数：{annotation_id!r}/{image_id!r}"
            )
        if annotation_id in annotation_ids:
            raise DatasetPreparationError(f"标注 ID 重复：{annotation_id!r}")
        annotation_ids.add(annotation_id)
        if image_id not in image_by_id:
            raise DatasetPreparationError(f"标注引用未知图片 ID：{image_id!r}")
        if annotation.get("category_id") != category_id:
            raise DatasetPreparationError(
                f"标注 {annotation_id!r} 使用了非 ship 类别：{annotation.get('category_id')!r}"
            )
        if annotation.get("iscrowd", 0):
            raise DatasetPreparationError(f"标注 {annotation_id!r} 是 iscrowd，不能静默转换。")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise DatasetPreparationError(f"标注 {annotation_id!r} 的 bbox 必须为 [x,y,w,h]。")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
            raise DatasetPreparationError(f"标注 {annotation_id!r} 的 bbox 含非法数值。")
        x, y, width, height = (float(value) for value in bbox)
        image = image_by_id[image_id]
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            raise DatasetPreparationError(f"标注 {annotation_id!r} 的 bbox 尺寸或坐标无效。")
        if x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6:
            raise DatasetPreparationError(f"标注 {annotation_id!r} 的 bbox 超出图片边界。")
        annotations_by_image[image_id].append(annotation)

    labels = {}
    image_files = {}
    for image_id, image in image_by_id.items():
        width = image["width"]
        height = image["height"]
        lines = []
        for annotation in sorted(
            annotations_by_image[image_id], key=lambda value: value["id"]
        ):
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            x_center = (x + box_width / 2.0) / width
            y_center = (y + box_height / 2.0) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            values = (x_center, y_center, normalized_width, normalized_height)
            if not all(0.0 <= value <= 1.0 for value in values):
                raise DatasetPreparationError(
                    f"标注 {annotation['id']!r} 转换后的 YOLO 坐标超出 [0,1]。"
                )
            lines.append("0 " + " ".join(f"{value:.10g}" for value in values))
        labels[Path(image["file_name"]).stem + ".txt"] = (
            "\n".join(lines) + ("\n" if lines else "")
        )
        image_files[image["file_name"]] = image_dir / image["file_name"]

    return {
        "annotation_path": annotation_path,
        "image_dir": image_dir,
        "image_files": image_files,
        "labels": labels,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "empty_image_count": sum(not value for value in labels.values()),
    }


def build_manifest(source_root: Path, split_data: dict[str, dict]) -> dict:
    return {
        "schema_version": 1,
        "format": "ultralytics-yolo-detect",
        "image_storage": "hardlink",
        "class_names": ["ship"],
        "source_root": source_root.name,
        "splits": {
            split: {
                "annotation_file": str(
                    split_data[split]["annotation_path"].relative_to(source_root)
                ),
                "annotation_sha256": sha256_file(split_data[split]["annotation_path"]),
                "image_directory": split,
                "image_count": split_data[split]["image_count"],
                "annotation_count": split_data[split]["annotation_count"],
                "empty_image_count": split_data[split]["empty_image_count"],
            }
            for split in SPLITS
        },
    }


def verify_existing(output_root: Path, split_data: dict[str, dict], manifest: dict) -> None:
    manifest_path = output_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise DatasetPreparationError(
            f"已有输出缺少清单，拒绝覆盖：{output_root}"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        existing_manifest = json.load(handle)
    if existing_manifest != manifest:
        raise DatasetPreparationError(
            f"已有输出与当前源数据哈希或计数不一致，拒绝覆盖：{output_root}"
        )

    for split in SPLITS:
        image_dir = output_root / "images" / split
        if not image_dir.is_dir() or image_dir.is_symlink():
            raise DatasetPreparationError(f"图片入口不是普通目录：{image_dir}")
        expected_images = split_data[split]["image_files"]
        actual_image_names = {path.name for path in image_dir.iterdir()}
        if actual_image_names != set(expected_images):
            raise DatasetPreparationError(f"图片硬链接集合不一致：{image_dir}")
        for filename, source_path in expected_images.items():
            if not os.path.samefile(image_dir / filename, source_path):
                raise DatasetPreparationError(f"图片不是源文件的硬链接：{image_dir / filename}")

        label_dir = output_root / "labels" / split
        expected_labels = split_data[split]["labels"]
        actual_names = {path.name for path in label_dir.glob("*.txt")} if label_dir.is_dir() else set()
        if actual_names != set(expected_labels):
            raise DatasetPreparationError(f"标签文件集合不一致：{label_dir}")
        for filename, expected_text in expected_labels.items():
            if (label_dir / filename).read_text(encoding="utf-8") != expected_text:
                raise DatasetPreparationError(f"标签内容不一致：{label_dir / filename}")


def create_output(output_root: Path, split_data: dict[str, dict], manifest: dict) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        for split in SPLITS:
            label_dir = temporary_root / "labels" / split
            label_dir.mkdir(parents=True)
            for filename, text in split_data[split]["labels"].items():
                (label_dir / filename).write_text(text, encoding="utf-8")

            image_dir = temporary_root / "images" / split
            image_dir.mkdir(parents=True)
            for filename, source_path in split_data[split]["image_files"].items():
                os.link(source_path, image_dir / filename)

        (temporary_root / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_root.rename(output_root)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def prepare_dataset(source_root: Path, output_root: Path, check_only: bool = False) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root != source_root / "yolo":
        raise DatasetPreparationError(
            f"输出目录必须是源 SAR 目录下的固定生成目录：{source_root / 'yolo'}"
        )

    split_data = {split: load_split(source_root, split) for split in SPLITS}
    manifest = build_manifest(source_root, split_data)
    if check_only:
        return manifest
    if output_root.exists():
        if not output_root.is_dir():
            raise DatasetPreparationError(f"输出路径已存在且不是目录：{output_root}")
        verify_existing(output_root, split_data, manifest)
        return manifest

    create_output(output_root, split_data, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY_ROOT / "SAR_COCO_ship",
        help="包含 annotations/train2017/val2017 的原始 SAR COCO 目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="YOLO 生成目录；默认使用 <source-root>/yolo。",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只校验源 COCO 数据并打印清单，不创建标签或图片硬链接。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = (args.output_root or source_root / "yolo").resolve()
    try:
        manifest = prepare_dataset(source_root, output_root, check_only=args.check_only)
    except (DatasetPreparationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"SAR YOLO 数据准备失败：{error}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if args.check_only:
        print("源 SAR COCO 数据校验通过；未写入任何文件。")
    elif output_root.exists():
        print(f"SAR YOLO 数据已准备并校验：{output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
