#!/usr/bin/env python3
"""将 Ultralytics predictions.json 映射回原始 SAR COCO ID 并统一评估。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl")


class YoloCocoEvaluationError(RuntimeError):
    """表示 YOLO 预测无法安全映射到原始 COCO 数据。"""


def load_json(path: Path):
    if not path.is_file():
        raise YoloCocoEvaluationError(f"文件不存在：{path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_image_maps(coco_data: dict) -> tuple[dict[str, int], dict[str, int], int]:
    categories = coco_data.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise YoloCocoEvaluationError("SAR COCO 标注必须且只能包含一个类别。")
    category = categories[0]
    if category.get("name") != "ship" or not isinstance(category.get("id"), int):
        raise YoloCocoEvaluationError("SAR COCO 唯一类别必须为整数 ID 的 ship。")

    images = coco_data.get("images")
    if not isinstance(images, list) or not images:
        raise YoloCocoEvaluationError("SAR COCO 标注缺少 images。")
    by_name = {}
    by_stem = {}
    for image in images:
        image_id = image.get("id")
        file_name = image.get("file_name")
        if not isinstance(image_id, int) or not isinstance(file_name, str):
            raise YoloCocoEvaluationError("COCO 图片必须具有整数 id 和字符串 file_name。")
        name = Path(file_name).name
        stem = Path(file_name).stem
        if name in by_name or stem in by_stem:
            raise YoloCocoEvaluationError(f"COCO 图片文件名或主名不唯一：{file_name}")
        by_name[name] = image_id
        by_stem[stem] = image_id
    return by_name, by_stem, category["id"]


def convert_predictions(coco_data: dict, predictions: list[dict]) -> list[dict]:
    if not isinstance(predictions, list):
        raise YoloCocoEvaluationError("Ultralytics predictions.json 顶层必须是列表。")
    by_name, by_stem, category_id = build_image_maps(coco_data)
    converted = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise YoloCocoEvaluationError(f"预测 {index} 不是映射。")

        file_name = prediction.get("file_name")
        if isinstance(file_name, str):
            image_id = by_name.get(Path(file_name).name)
        else:
            source_image_id = prediction.get("image_id")
            image_id = by_stem.get(str(source_image_id))
        if image_id is None:
            raise YoloCocoEvaluationError(
                f"预测 {index} 无法通过 file_name/image_id 映射到原始 COCO 图片。"
            )

        source_category_id = prediction.get("category_id")
        if source_category_id not in {0, category_id}:
            raise YoloCocoEvaluationError(
                f"预测 {index} 含非 ship 类别 ID：{source_category_id!r}"
            )

        bbox = prediction.get("bbox")
        score = prediction.get("score")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise YoloCocoEvaluationError(f"预测 {index} 的 bbox 必须为 [x,y,w,h]。")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
            raise YoloCocoEvaluationError(f"预测 {index} 的 bbox 含非法数值。")
        if float(bbox[2]) < 0 or float(bbox[3]) < 0:
            raise YoloCocoEvaluationError(f"预测 {index} 的 bbox 宽高不能为负。")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise YoloCocoEvaluationError(f"预测 {index} 的 score 无效。")
        if not 0.0 <= float(score) <= 1.0:
            raise YoloCocoEvaluationError(f"预测 {index} 的 score 超出 [0,1]。")

        converted.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(value) for value in bbox],
                "score": float(score),
            }
        )
    if not converted:
        raise YoloCocoEvaluationError("预测列表为空，无法执行 COCOeval。")
    return converted


def run_coco_evaluation(
    annotations_path: Path, converted_predictions: list[dict]
) -> tuple[dict[str, float], str]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise YoloCocoEvaluationError(
            "当前环境缺少 pycocotools；请激活 Difdet 环境后重试。"
        ) from error

    with tempfile.TemporaryDirectory(prefix="yolo-coco-eval-") as temporary:
        prediction_path = Path(temporary) / "predictions.json"
        prediction_path.write_text(
            json.dumps(converted_predictions, ensure_ascii=False), encoding="utf-8"
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            coco_ground_truth = COCO(str(annotations_path))
            coco_predictions = coco_ground_truth.loadRes(str(prediction_path))
            evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
            evaluator.params.imgIds = sorted(coco_ground_truth.getImgIds())
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        summary = output.getvalue()
        metrics = {
            name: float(evaluator.stats[index])
            for index, name in enumerate(METRIC_NAMES)
        }
    return metrics, summary


def evaluate_predictions(
    annotations_path: Path,
    predictions_path: Path,
    output_root: Path | None = None,
    timestamp: str | None = None,
) -> Path:
    annotations_path = annotations_path.resolve()
    predictions_path = predictions_path.resolve()
    coco_data = load_json(annotations_path)
    raw_predictions = load_json(predictions_path)
    converted = convert_predictions(coco_data, raw_predictions)
    metrics, summary = run_coco_evaluation(annotations_path, converted)

    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (output_root or predictions_path.parent / "evaluations").resolve()
    output_dir = output_root / timestamp
    if output_dir.exists():
        raise YoloCocoEvaluationError(f"评估输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True)

    (output_dir / "coco_instances_results.json").write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "schema_version": 1,
        "evaluator": "pycocotools.COCOeval",
        "iou_type": "bbox",
        "max_dets": [1, 10, 100],
        "annotations": str(annotations_path),
        "predictions": str(predictions_path),
        "prediction_count": len(converted),
        "metrics": metrics,
    }
    (output_dir / "coco_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "coco_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"COCOeval 结果已写入：{output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPOSITORY_ROOT / "SAR_COCO_ship/annotations/instances_val2017.json",
        help="原始 SAR COCO 验证标注。",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Ultralytics 在 best.pt 最终验证后生成的 predictions.json。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="评估根目录；默认在 predictions.json 同级 evaluations/ 下创建 UTC 时间戳目录。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evaluate_predictions(args.annotations, args.predictions, args.output_root)
    except (YoloCocoEvaluationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"YOLO COCOeval 失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
