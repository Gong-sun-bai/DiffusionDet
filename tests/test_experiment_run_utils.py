import json
import tempfile
import unittest
from pathlib import Path

from tools.experiment_run_utils import (
    LoadedExperiment,
    action_for,
    load_experiment,
)


class ActionTests(unittest.TestCase):
    def experiment(self, root):
        return LoadedExperiment(
            config=root / "config.yaml",
            run_dir=root / "run",
            config_hash="a" * 64,
            max_iter=100,
        )

    def test_train_resume_skip_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self.experiment(root)
            self.assertEqual(action_for(experiment), "train")
            experiment.run_dir.mkdir()
            (experiment.run_dir / "model_latest.pth").write_bytes(b"x")
            (experiment.run_dir / "last_checkpoint").write_text(
                "model_latest.pth", encoding="utf-8"
            )
            manifest_path = experiment.run_dir / "experiment_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "paused",
                        "execution": {"config_sha256": "a" * 64},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(action_for(experiment), "resume")

            (experiment.run_dir / "result_summary.json").write_text(
                "{}", encoding="utf-8"
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "execution": {"config_sha256": "a" * 64},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(action_for(experiment), "skip")

            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "execution": {"config_sha256": "b" * 64},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                action_for(experiment)

    def test_formal_configs_resolve_as_independent_experiments(self):
        root = Path(__file__).resolve().parents[1]
        repvit = load_experiment(
            root
            / "configs/experiments/sar_ship/"
            / "sar-015-under10m-repvit-full-seed40244023.yaml"
        )
        mobilenet = load_experiment(
            root
            / "configs/experiments/sar_ship/"
            / "sar-016-under10m-timm-mnv4-full-seed40244023.yaml"
        )
        self.assertEqual(repvit.max_iter, 254264)
        self.assertEqual(mobilenet.max_iter, 254264)
        self.assertNotEqual(repvit.run_dir, mobilenet.run_dir)


if __name__ == "__main__":
    unittest.main()
