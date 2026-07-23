import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from experiment_manager import (
    ExperimentError,
    configure_experiment,
    finish_experiment,
    initialize_experiment,
    resolve_run_dir,
    validate_experiment_id,
    validate_output_state,
    validate_slug,
)


def make_cfg(**overrides):
    cfg = {
        "EXPERIMENT": {
            "ID": "sar-005",
            "NAME": "r18-fpn128-seed40244023",
            "DATASET": "sar_ship",
            "TRACK": "accuracy",
            "KIND": "baseline",
            "BASELINE": "",
            "PURPOSE": "建立固定种子基线",
            "HYPOTHESIS": "固定种子后结果可重复",
            "CHANGES": [],
            "OUTPUT_ROOT": "./runs",
        },
        "SEED": 40244023,
        "MODEL": {
            "WEIGHTS": "",
            "BACKBONE": {"NAME": "build_resnet_fpn_backbone"},
            "RESNETS": {"DEPTH": 18},
            "FPN": {"OUT_CHANNELS": 128},
            "DiffusionDet": {
                "HIDDEN_DIM": 128,
                "NUM_PROPOSALS": 500,
                "SAMPLE_STEP": 1,
            },
        },
        "SOLVER": {
            "IMS_PER_BATCH": 16,
            "BASE_LR": 0.000025,
            "MAX_ITER": 397288,
            "STEPS": (278102, 357559),
        },
        "DATASETS": {
            "TRAIN": ("sar_ship_train",),
            "TEST": ("sar_ship_val",),
        },
    }
    for dotted_key, value in overrides.items():
        node = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def make_args(**overrides):
    values = {
        "config_file": "configs/experiments/sar_ship/sar-005.yaml",
        "eval_only": False,
        "resume": False,
        "opts": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ValidationTests(unittest.TestCase):
    def test_valid_and_invalid_experiment_ids(self):
        validate_experiment_id("sar-005")
        validate_experiment_id("panda-000")
        for value in ("SAR-005", "sar-5", "sar_005", "sar-0005"):
            with self.subTest(value=value), self.assertRaises(ExperimentError):
                validate_experiment_id(value)

    def test_valid_and_invalid_slugs(self):
        validate_slug("r18-fpn128-seed40244023")
        for value in ("R18", "r18_fpn", "r18--fpn", "r18 fpn"):
            with self.subTest(value=value), self.assertRaises(ExperimentError):
                validate_slug(value)

    def test_nonhistorical_requires_fixed_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ExperimentError, "SEED"):
                configure_experiment(
                    make_cfg(**{"SEED": -1}), make_args(), Path(temporary)
                )

    def test_experiment_id_prefix_matches_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ExperimentError, "前缀"):
                configure_experiment(
                    make_cfg(**{"EXPERIMENT.ID": "panda-005"}),
                    make_args(),
                    Path(temporary),
                )

    def test_milestones_must_precede_max_iter(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ExperimentError, "STEPS"):
                configure_experiment(
                    make_cfg(**{"SOLVER.STEPS": (278102, 500000)}),
                    make_args(),
                    Path(temporary),
                )

    def test_ablation_requires_baseline_and_structured_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = make_cfg(
                **{
                    "EXPERIMENT.KIND": "ablation",
                    "EXPERIMENT.BASELINE": "sar-005",
                    "EXPERIMENT.CHANGES": [
                        "MODEL.DiffusionDet.NUM_PROPOSALS: 500 -> 300"
                    ],
                }
            )
            configure_experiment(cfg, make_args(), Path(temporary))
            cfg["EXPERIMENT"]["CHANGES"] = ["proposals changed"]
            with self.assertRaisesRegex(ExperimentError, "CHANGES"):
                configure_experiment(cfg, make_args(), Path(temporary))


class DirectorySafetyTests(unittest.TestCase):
    def test_run_directory_is_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = configure_experiment(make_cfg(), make_args(), root)
            self.assertEqual(
                context.run_dir,
                root
                / "runs"
                / "sar_ship"
                / "sar-005__r18-fpn128-seed40244023",
            )
            self.assertEqual(context.run_dir, resolve_run_dir(root, context.metadata))

    def test_fresh_training_rejects_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = configure_experiment(make_cfg(), make_args(), Path(temporary))
            context.output_dir.mkdir(parents=True)
            (context.output_dir / "existing.txt").write_text("occupied")
            with self.assertRaisesRegex(ExperimentError, "非空目录"):
                validate_output_state(context)

    def test_resume_requires_valid_checkpoint_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = configure_experiment(
                make_cfg(), make_args(resume=True), root
            )
            context.run_dir.mkdir(parents=True)
            with self.assertRaisesRegex(ExperimentError, "last_checkpoint"):
                validate_output_state(context)
            (context.run_dir / "last_checkpoint").write_text("model_final.pth")
            with self.assertRaisesRegex(ExperimentError, "不存在"):
                validate_output_state(context)
            (context.run_dir / "model_final.pth").write_bytes(b"checkpoint")
            validate_output_state(context)

    def test_historical_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = make_cfg(
                **{
                    "EXPERIMENT.ID": "sar-001",
                    "EXPERIMENT.KIND": "historical",
                    "SEED": -1,
                }
            )
            with self.assertRaisesRegex(ExperimentError, "历史配置"):
                configure_experiment(cfg, make_args(), Path(temporary))

    def test_evaluation_gets_timestamped_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = make_cfg(**{"MODEL.WEIGHTS": "weights/model_final.pth"})
            run_dir = (
                root
                / "runs"
                / "sar_ship"
                / "sar-005__r18-fpn128-seed40244023"
            )
            run_dir.mkdir(parents=True)
            (root / "weights").mkdir()
            (root / "weights/model_final.pth").write_bytes(b"checkpoint")
            now = datetime(2026, 7, 23, 1, 2, 3, 456789, tzinfo=timezone.utc)
            context = configure_experiment(
                cfg,
                make_args(
                    eval_only=True,
                    opts=["MODEL.WEIGHTS", "weights/model_final.pth"],
                ),
                root,
                now=now,
            )
            self.assertEqual(
                context.output_dir,
                run_dir / "evaluations" / "20260723T010203.456789Z",
            )
            validate_output_state(context)

    def test_evaluation_rejects_prediction_file_as_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = make_cfg(
                **{"MODEL.WEIGHTS": "inference/instances_predictions.pth"}
            )
            with self.assertRaisesRegex(ExperimentError, "预测结果"):
                configure_experiment(
                    cfg,
                    make_args(
                        eval_only=True,
                        opts=[
                            "MODEL.WEIGHTS",
                            "inference/instances_predictions.pth",
                        ],
                    ),
                    Path(temporary),
                )

    def test_eval_only_and_resume_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ExperimentError, "不能同时"):
                configure_experiment(
                    make_cfg(),
                    make_args(eval_only=True, resume=True),
                    Path(temporary),
                )


class ManifestTests(unittest.TestCase):
    def test_manifest_and_summary_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = configure_experiment(make_cfg(), make_args(), Path(temporary))
            initialize_experiment(context, make_cfg())
            manifest_path = context.output_dir / "experiment_manifest.json"
            self.assertEqual(
                json.loads(manifest_path.read_text())["status"], "running"
            )
            finish_experiment(context, results={"bbox": {"AP": 66.9}})
            self.assertEqual(
                json.loads(manifest_path.read_text())["status"], "completed"
            )
            summary = json.loads(
                (context.output_dir / "result_summary.json").read_text()
            )
            self.assertEqual(summary["results"]["bbox"]["AP"], 66.9)


if __name__ == "__main__":
    unittest.main()
