import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from detectron2.utils.events import EventWriter
from diffusiondet.training_visualization import (
    TrainingPlotHook,
    TrainingPlotWriter,
    TrainingVisualizationError,
    generate_results_plot,
    load_metrics_history,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPOSITORY_ROOT / "tools" / "generate_training_plots.py"


def write_metrics(path, rows, *, trailing=""):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + trailing,
        encoding="utf-8",
    )


def sample_training_row(iteration):
    return {
        "iteration": iteration,
        "total_loss": 10.0 / (iteration + 1),
        "loss_ce": 1.0,
        "loss_bbox": 2.0,
        "loss_giou": 3.0,
        "lr": 2.5e-5,
        "time": 0.5,
        "data_time": 0.05,
    }


class MetricsParsingTests(unittest.TestCase):
    def test_merges_same_iteration_and_ignores_only_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "metrics.json"
            write_metrics(
                metrics_path,
                [
                    sample_training_row(19),
                    {"iteration": 19, "bbox/AP": 50.0},
                    sample_training_row(39),
                ],
                trailing='{"iteration": 59, "total_loss": ',
            )
            with self.assertWarnsRegex(RuntimeWarning, "最后一条"):
                history = load_metrics_history(metrics_path)

            self.assertEqual([row["iteration"] for row in history.rows], [19, 39])
            self.assertEqual(history.rows[0]["bbox/AP"], 50.0)
            self.assertEqual(history.training_records, 2)
            self.assertEqual(history.validation_records, 1)

    def test_rejects_malformed_internal_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "metrics.json"
            metrics_path.write_text(
                json.dumps(sample_training_row(19))
                + "\n"
                + "{broken}\n"
                + json.dumps(sample_training_row(39))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TrainingVisualizationError, "第 2 行"):
                load_metrics_history(metrics_path)

    def test_sorts_resume_records_and_filters_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "metrics.json"
            write_metrics(
                metrics_path,
                [
                    sample_training_row(200),
                    {"iteration": 100, "total_loss": float("nan"), "lr": 1e-5},
                    sample_training_row(300),
                ],
            )
            history = load_metrics_history(metrics_path)
            x_values, y_values = history.series("total_loss")
            self.assertEqual(x_values.tolist(), [200, 300])
            self.assertEqual(len(y_values), 2)
            self.assertEqual(
                [row["iteration"] for row in history.rows],
                [100, 200, 300],
            )

    def test_rejects_missing_drawable_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "metrics.json"
            write_metrics(metrics_path, [{"iteration": 19, "eta_seconds": 12.0}])
            with self.assertRaisesRegex(TrainingVisualizationError, "没有可绘制"):
                load_metrics_history(metrics_path)


class RenderingTests(unittest.TestCase):
    def _make_run(self, root, name, rows):
        run_dir = root / name
        run_dir.mkdir()
        write_metrics(run_dir / "metrics.json", rows)
        (run_dir / "config.yaml").write_text(
            "SOLVER:\n  MAX_ITER: 1000\n",
            encoding="utf-8",
        )
        return run_dir

    def _assert_valid_png(self, path):
        self.assertTrue(path.is_file())
        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (2400, 1350))
            image.verify()

    def test_renders_supported_training_and_validation_shapes(self):
        scenarios = {
            "full": [
                sample_training_row(19),
                sample_training_row(199),
                {
                    "iteration": 199,
                    "bbox/AP": 50.0,
                    "bbox/AP50": 80.0,
                    "bbox/AP75": 55.0,
                    "bbox/APs": 40.0,
                    "bbox/APm": 55.0,
                    "bbox/APl": 60.0,
                },
                sample_training_row(399),
                {"iteration": 399, "bbox/AP": 52.0},
            ],
            "final-ap-only": [
                sample_training_row(19),
                sample_training_row(999),
                {
                    "iteration": 1000,
                    "bbox/AP": 51.0,
                    "bbox/AP50": 82.0,
                    "bbox/AP75": 56.0,
                    "bbox/APs": 41.0,
                    "bbox/APm": 57.0,
                    "bbox/APl": 62.0,
                },
            ],
            "no-validation": [sample_training_row(19), sample_training_row(199)],
            "single-smoke": [sample_training_row(19)],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, rows in scenarios.items():
                with self.subTest(name=name):
                    run_dir = self._make_run(root, name, rows)
                    summary = generate_results_plot(run_dir)
                    self._assert_valid_png(summary.output_path)

    def test_repeated_render_replaces_existing_file_and_leaves_no_temp_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._make_run(root, "overwrite", [sample_training_row(19)])
            output = run_dir / "results.png"
            output.write_bytes(b"not a png")
            generate_results_plot(run_dir)
            self._assert_valid_png(output)
            self.assertEqual(list(run_dir.glob(".results.*.tmp.png")), [])


class CliTests(unittest.TestCase):
    def test_cli_generates_results_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            write_metrics(
                run_dir / "metrics.json",
                [sample_training_row(19), {"iteration": 20, "bbox/AP": 42.0}],
            )
            completed = subprocess.run(
                [sys.executable, str(CLI_PATH), str(run_dir)],
                cwd=REPOSITORY_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("训练记录=1", completed.stdout)
            self.assertIn("验证记录=1", completed.stdout)
            self.assertTrue((run_dir / "results.png").is_file())

    def test_cli_returns_nonzero_for_missing_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            completed = subprocess.run(
                [sys.executable, str(CLI_PATH), str(run_dir)],
                cwd=REPOSITORY_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("指标文件不存在", completed.stderr)


class FakeWriter(EventWriter):
    def __init__(self):
        self.write_count = 0
        self.close_count = 0

    def write(self):
        self.write_count += 1

    def close(self):
        self.close_count += 1


class WriterAndHookTests(unittest.TestCase):
    def test_hook_refreshes_at_period_evaluation_and_training_end(self):
        writer = FakeWriter()
        hook = TrainingPlotHook(writer, period=200, eval_period=500)
        hook.trainer = SimpleNamespace(iter=198, max_iter=1000)
        hook.after_step()
        self.assertEqual(writer.write_count, 0)

        hook.trainer.iter = 199
        hook.after_step()
        hook.trainer.iter = 499
        hook.after_step()
        self.assertEqual(writer.write_count, 2)

        hook.after_train()
        self.assertEqual(writer.write_count, 3)
        self.assertEqual(writer.close_count, 1)

    def test_writer_render_error_is_nonfatal(self):
        calls = []

        def failing_render(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("synthetic render failure")

        with tempfile.TemporaryDirectory() as temporary:
            writer = TrainingPlotWriter(temporary, render=failing_render)
            with self.assertLogs(
                "diffusiondet.training_visualization",
                level="WARNING",
            ) as captured:
                writer.write()
                writer.write()
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(captured.output), 1)

    def test_hook_rejects_nonpositive_period(self):
        with self.assertRaisesRegex(ValueError, "PERIOD"):
            TrainingPlotHook(FakeWriter(), period=0)


if __name__ == "__main__":
    unittest.main()
