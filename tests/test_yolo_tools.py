import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_yolo_coco import (
    YoloCocoEvaluationError,
    evaluate_predictions,
)
from tools.prepare_sar_yolo_dataset import (
    DatasetPreparationError,
    prepare_dataset,
)


HAS_PYCOCOTOOLS = importlib.util.find_spec("pycocotools") is not None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PrepareSarYoloDatasetTests(unittest.TestCase):
    def make_source(self, root):
        source = root / "SAR_COCO_ship"
        categories = [{"id": 1, "name": "ship", "supercategory": "none"}]
        split_values = {
            "train2017": (
                [
                    {"id": 1, "file_name": "train-a.jpg", "width": 100, "height": 50},
                    {"id": 2, "file_name": "train-empty.jpg", "width": 64, "height": 64},
                ],
                [
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 30, 20],
                        "area": 600,
                        "iscrowd": 0,
                    }
                ],
            ),
            "val2017": (
                [{"id": 3, "file_name": "val-a.jpg", "width": 80, "height": 40}],
                [
                    {
                        "id": 11,
                        "image_id": 3,
                        "category_id": 1,
                        "bbox": [8, 4, 16, 8],
                        "area": 128,
                        "iscrowd": 0,
                    }
                ],
            ),
        }
        for split, (images, annotations) in split_values.items():
            image_dir = source / split
            image_dir.mkdir(parents=True)
            for image in images:
                (image_dir / image["file_name"]).write_bytes(b"image")
            write_json(
                source / "annotations" / f"instances_{split}.json",
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                },
            )
        return source

    def test_conversion_is_correct_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.make_source(Path(temporary))
            output = source / "yolo"
            manifest = prepare_dataset(source, output)

            self.assertEqual(manifest["splits"]["train2017"]["image_count"], 2)
            self.assertEqual(manifest["splits"]["train2017"]["empty_image_count"], 1)
            self.assertEqual(
                (output / "labels/train2017/train-a.txt").read_text(encoding="utf-8"),
                "0 0.25 0.4 0.3 0.4\n",
            )
            self.assertEqual(
                (output / "labels/train2017/train-empty.txt").read_text(encoding="utf-8"),
                "",
            )
            generated_image = output / "images/train2017/train-a.jpg"
            self.assertTrue(generated_image.is_file())
            self.assertFalse(generated_image.is_symlink())
            self.assertTrue(generated_image.samefile(source / "train2017/train-a.jpg"))

            repeated = prepare_dataset(source, output)
            self.assertEqual(repeated, manifest)
            self.assertEqual(
                sorted(path.name for path in output.parent.glob(".yolo.tmp-*")), []
            )

    def test_check_only_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.make_source(Path(temporary))
            manifest = prepare_dataset(source, source / "yolo", check_only=True)
            self.assertEqual(manifest["splits"]["val2017"]["annotation_count"], 1)
            self.assertFalse((source / "yolo").exists())

    def test_existing_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.make_source(Path(temporary))
            output = source / "yolo"
            prepare_dataset(source, output)
            (output / "labels/train2017/train-a.txt").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaises(DatasetPreparationError):
                prepare_dataset(source, output)


@unittest.skipUnless(HAS_PYCOCOTOOLS, "需要 pycocotools")
class EvaluateYoloCocoTests(unittest.TestCase):
    def test_filename_and_category_are_mapped_before_cocoeval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "instances_val2017.json"
            predictions = root / "predictions.json"
            write_json(
                annotations,
                {
                    "info": {},
                    "licenses": [],
                    "images": [
                        {"id": 42, "file_name": "sample.jpg", "width": 100, "height": 100}
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 42,
                            "category_id": 1,
                            "bbox": [10, 10, 20, 20],
                            "area": 400,
                            "iscrowd": 0,
                        }
                    ],
                    "categories": [{"id": 1, "name": "ship", "supercategory": "none"}],
                },
            )
            write_json(
                predictions,
                [
                    {
                        "image_id": "sample",
                        "file_name": "sample.jpg",
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "score": 0.99,
                    }
                ],
            )

            output = evaluate_predictions(
                annotations, predictions, root / "evaluations", timestamp="fixed"
            )
            result = json.loads((output / "coco_metrics.json").read_text(encoding="utf-8"))
            converted = json.loads(
                (output / "coco_instances_results.json").read_text(encoding="utf-8")
            )
            self.assertAlmostEqual(result["metrics"]["AP"], 1.0)
            self.assertAlmostEqual(result["metrics"]["AP50"], 1.0)
            self.assertAlmostEqual(result["metrics"]["AP75"], 1.0)
            self.assertEqual(converted[0]["image_id"], 42)
            self.assertEqual(converted[0]["category_id"], 1)
            with self.assertRaises(YoloCocoEvaluationError):
                evaluate_predictions(
                    annotations, predictions, root / "evaluations", timestamp="fixed"
                )


if __name__ == "__main__":
    unittest.main()
