"""Project dataset registration with repository-root-based paths."""

from pathlib import Path

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances


DATASETS = {
    "lpi_train": ("LPI_COCO/annotations/instances_train.json", "LPI_COCO"),
    "lpi_val": ("LPI_COCO/annotations/instances_val.json", "LPI_COCO"),
    "lpi_test": ("LPI_COCO/annotations/instances_test.json", "LPI_COCO"),
    "sar_ship_train": (
        "SAR_COCO_ship/annotations/instances_train2017.json",
        "SAR_COCO_ship/train2017",
    ),
    "sar_ship_val": (
        "SAR_COCO_ship/annotations/instances_val2017.json",
        "SAR_COCO_ship/val2017",
    ),
    "panda_yolo_train": (
        "panda_coco_data/annotations/instances_train2017.json",
        "panda_coco_data/train2017",
    ),
    "panda_yolo_val": (
        "panda_coco_data/annotations/instances_val2017.json",
        "panda_coco_data/val2017",
    ),
}


def register_project_datasets(repository_root):
    """Register SAR, PANDA and LPI datasets using absolute local paths."""
    repository_root = Path(repository_root).resolve()
    registered = set(DatasetCatalog.list())
    for name, (json_relpath, image_relpath) in DATASETS.items():
        if name in registered:
            continue
        register_coco_instances(
            name,
            {},
            str(repository_root / json_relpath),
            str(repository_root / image_relpath),
        )
