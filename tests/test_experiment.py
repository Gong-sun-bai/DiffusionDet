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
    validate_checkpoint_policy,
    validate_experiment_id,
    validate_output_state,
    validate_slug,
    validate_training_plot_policy,
)


def make_cfg(**overrides):
    cfg = {
        "EXPERIMENT": {
            "ID": "sar-005",
            "NAME": "r18-fpn128-seed40244023",
            "DATASET": "sar_ship",
            "DESCRIPTION": "固定种子 R18/FPN128 基线",
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
            "WARMUP_ITERS": 1000,
            "CHECKPOINT_PERIOD": 5000,
            "CHECKPOINT_RETENTION": "all",
            "AMP": {"ENABLED": False},
        },
        "TEST": {
            "EVAL_PERIOD": 0,
            "BEST_CHECKPOINT": {
                "ENABLED": False,
                "METRIC": "bbox/AP",
                "MODE": "max",
            },
        },
        "TRAINING_PLOTS": {
            "ENABLED": True,
            "PERIOD": 200,
        },
        "DATALOADER": {"NUM_WORKERS": 4},
        "INPUT": {
            "MIN_SIZE_TRAIN": (256, 384, 512, 640, 768, 896, 1024),
            "MAX_SIZE_TRAIN": 1024,
            "MIN_SIZE_TEST": 256,
            "MAX_SIZE_TEST": 1024,
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
        "run_until_iter": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ValidationTests(unittest.TestCase):
    def test_run_until_iter_is_bounded_by_full_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = configure_experiment(
                make_cfg(),
                make_args(run_until_iter=20000),
                Path(temporary),
            )
            self.assertEqual(context.run_until_iter, 20000)
            with self.assertRaisesRegex(ExperimentError, "run-until-iter"):
                configure_experiment(
                    make_cfg(),
                    make_args(run_until_iter=500000),
                    Path(temporary),
                )

    def test_eval_only_rejects_run_until_iter(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ExperimentError, "run-until-iter"):
                configure_experiment(
                    make_cfg(),
                    make_args(eval_only=True, run_until_iter=200),
                    Path(temporary),
                )

    def test_valid_and_invalid_experiment_ids(self):
        validate_experiment_id("sar-005")
        validate_experiment_id("sar-test-001")
        validate_experiment_id("panda-000")
        for value in (
            "SAR-005",
            "sar-5",
            "sar_005",
            "sar-0005",
            "sar-smoke-001",
            "sar-test-01",
        ):
            with self.subTest(value=value), self.assertRaises(ExperimentError):
                validate_experiment_id(value)

    def test_valid_and_invalid_slugs(self):
        validate_slug("r18-fpn128-seed40244023")
        for value in ("R18", "r18_fpn", "r18--fpn", "r18 fpn"):
            with self.subTest(value=value), self.assertRaises(ExperimentError):
                validate_slug(value)

    def test_training_requires_fixed_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ExperimentError, "SEED"):
                configure_experiment(
                    make_cfg(**{"SEED": -1}), make_args(), Path(temporary)
                )

    def test_resume_requires_fixed_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ExperimentError, "SEED"):
                configure_experiment(
                    make_cfg(**{"SEED": -1}),
                    make_args(resume=True),
                    Path(temporary),
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

    def test_description_is_optional(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = make_cfg()
            del cfg["EXPERIMENT"]["DESCRIPTION"]
            context = configure_experiment(
                cfg,
                make_args(),
                Path(temporary),
            )
            self.assertEqual(context.metadata.description, "")

    def test_test_id_does_not_require_separate_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = make_cfg(
                **{
                    "EXPERIMENT.ID": "sar-test-001",
                    "EXPERIMENT.DESCRIPTION": "20 iter 训练链路测试",
                    "SOLVER.MAX_ITER": 20,
                    "SOLVER.STEPS": (),
                }
            )
            context = configure_experiment(cfg, make_args(), Path(temporary))
            self.assertEqual(context.metadata.experiment_id, "sar-test-001")
            self.assertEqual(context.metadata.description, "20 iter 训练链路测试")

    def test_evaluation_allows_legacy_training_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = make_cfg(
                **{
                    "SEED": -1,
                    "SOLVER.MAX_ITER": 99400,
                    "SOLVER.STEPS": (278102, 357559),
                    "MODEL.WEIGHTS": "weights/model_final.pth",
                }
            )
            context = configure_experiment(
                cfg,
                make_args(
                    eval_only=True,
                    opts=["MODEL.WEIGHTS", "weights/model_final.pth"],
                ),
                Path(temporary),
            )
            self.assertEqual(context.mode, "evaluation")

    def test_checkpoint_policy_requires_supported_values(self):
        with self.assertRaisesRegex(ExperimentError, "RETENTION"):
            validate_checkpoint_policy(
                make_cfg(**{"SOLVER.CHECKPOINT_RETENTION": "best-only"})
            )
        with self.assertRaisesRegex(ExperimentError, "CHECKPOINT_PERIOD"):
            validate_checkpoint_policy(
                make_cfg(
                    **{
                        "SOLVER.CHECKPOINT_RETENTION": "latest",
                        "SOLVER.CHECKPOINT_PERIOD": 0,
                    }
                )
            )
        with self.assertRaisesRegex(ExperimentError, "EVAL_PERIOD"):
            validate_checkpoint_policy(
                make_cfg(**{"TEST.BEST_CHECKPOINT.ENABLED": True})
            )
        validate_checkpoint_policy(
            make_cfg(
                **{
                    "SOLVER.CHECKPOINT_RETENTION": "latest",
                    "TEST.EVAL_PERIOD": 5000,
                    "TEST.BEST_CHECKPOINT.ENABLED": True,
                }
            )
        )

    def test_training_plot_policy_requires_positive_period_when_enabled(self):
        with self.assertRaisesRegex(ExperimentError, "PERIOD"):
            validate_training_plot_policy(
                make_cfg(**{"TRAINING_PLOTS.PERIOD": 0})
            )
        validate_training_plot_policy(
            make_cfg(
                **{
                    "TRAINING_PLOTS.ENABLED": False,
                    "TRAINING_PLOTS.PERIOD": 0,
                }
            )
        )


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

    def test_fresh_training_returns_nonempty_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = configure_experiment(make_cfg(), make_args(), Path(temporary))
            context.output_dir.mkdir(parents=True)
            (context.output_dir / "existing.txt").write_text("occupied")
            self.assertEqual(
                validate_output_state(context), context.output_dir.resolve()
            )
            with self.assertRaisesRegex(ExperimentError, "已经存在"):
                initialize_experiment(context, make_cfg())

    def test_fresh_training_allows_missing_or_empty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = configure_experiment(make_cfg(), make_args(), Path(temporary))
            self.assertIsNone(validate_output_state(context))
            context.output_dir.mkdir(parents=True)
            self.assertIsNone(validate_output_state(context))

    def test_fresh_training_rejects_output_path_that_is_a_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = configure_experiment(make_cfg(), make_args(), Path(temporary))
            context.output_dir.parent.mkdir(parents=True)
            context.output_dir.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ExperimentError, "不是目录"):
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

    def test_legacy_training_is_rejected_by_seed_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = make_cfg(
                **{
                    "EXPERIMENT.ID": "sar-001",
                    "SEED": -1,
                }
            )
            with self.assertRaisesRegex(ExperimentError, "SEED"):
                configure_experiment(cfg, make_args(), Path(temporary))

    def test_existing_legacy_training_is_reported_before_seed_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = make_cfg(
                **{
                    "EXPERIMENT.ID": "sar-001",
                    "SEED": -1,
                }
            )
            run_dir = (
                root
                / "runs/sar_ship"
                / "sar-001__r18-fpn128-seed40244023"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
            context = configure_experiment(cfg, make_args(), root)
            self.assertEqual(validate_output_state(context), run_dir.resolve())

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

    def test_evaluation_accepts_latest_and_best_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = (
                root
                / "runs"
                / "sar_ship"
                / "sar-005__r18-fpn128-seed40244023"
            )
            run_dir.mkdir(parents=True)
            (root / "weights").mkdir()
            for filename in ("model_latest.pth", "model_best.pth"):
                with self.subTest(filename=filename):
                    weight_path = root / "weights" / filename
                    weight_path.write_bytes(b"checkpoint")
                    cfg = make_cfg(
                        **{"MODEL.WEIGHTS": f"weights/{filename}"}
                    )
                    context = configure_experiment(
                        cfg,
                        make_args(
                            eval_only=True,
                            opts=["MODEL.WEIGHTS", f"weights/{filename}"],
                        ),
                        root,
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
    def test_resume_preserves_screening_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = make_cfg()
            first = configure_experiment(
                cfg,
                make_args(run_until_iter=200),
                root,
            )
            initialize_experiment(first, cfg)
            finish_experiment(first, status="paused")
            manifest_path = first.output_dir / "experiment_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["screening_history"] = [
                {"status": "promoted", "reason": "passed 200 iter"}
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (first.output_dir / "model_latest.pth").write_bytes(b"checkpoint")
            (first.output_dir / "last_checkpoint").write_text(
                "model_latest.pth", encoding="utf-8"
            )

            resumed = configure_experiment(
                cfg,
                make_args(resume=True, run_until_iter=5000),
                root,
            )
            resumed_manifest = initialize_experiment(resumed, cfg)

            self.assertEqual(
                resumed_manifest["screening_history"],
                [{"status": "promoted", "reason": "passed 200 iter"}],
            )

    def test_paused_lifecycle_does_not_write_completed_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = configure_experiment(
                make_cfg(),
                make_args(run_until_iter=200),
                Path(temporary),
            )
            initialize_experiment(context, make_cfg())
            finish_experiment(context, status="paused")
            manifest = json.loads(
                (context.output_dir / "experiment_manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "paused")
            self.assertFalse((context.output_dir / "result_summary.json").exists())

    def test_manifest_and_summary_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = configure_experiment(make_cfg(), make_args(), Path(temporary))
            initialize_experiment(context, make_cfg())
            manifest_path = context.output_dir / "experiment_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(
                manifest["experiment"],
                {
                    "id": "sar-005",
                    "name": "r18-fpn128-seed40244023",
                    "dataset": "sar_ship",
                    "description": "固定种子 R18/FPN128 基线",
                },
            )
            self.assertEqual(
                manifest["parameters"]["DATALOADER.NUM_WORKERS"], 4
            )
            self.assertFalse(
                manifest["parameters"]["SOLVER.AMP.ENABLED"]
            )
            self.assertEqual(len(manifest["execution"]["config_sha256"]), 64)
            finish_experiment(context, results={"bbox": {"AP": 66.9}})
            self.assertEqual(
                json.loads(manifest_path.read_text())["status"], "completed"
            )
            summary = json.loads(
                (context.output_dir / "result_summary.json").read_text()
            )
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["results"]["bbox"]["AP"], 66.9)

    def test_blank_description_is_null_in_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = make_cfg(**{"EXPERIMENT.DESCRIPTION": ""})
            context = configure_experiment(cfg, make_args(), Path(temporary))
            manifest = initialize_experiment(context, cfg)
            self.assertIsNone(manifest["experiment"]["description"])


if __name__ == "__main__":
    unittest.main()
