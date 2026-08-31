import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.experiment_run_utils import LoadedExperiment
from tools.set_screening_status import set_status


class ScreeningStatusTests(unittest.TestCase):
    def test_records_reason_and_preserves_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            config = root / "config.yaml"
            config.write_text("EXPERIMENT: {}\n", encoding="utf-8")
            manifest_path = run_dir / "experiment_manifest.json"
            execution = {"config_sha256": "hash", "run_until_iter": 5000}
            manifest_path.write_text(
                json.dumps({"status": "paused", "execution": execution}),
                encoding="utf-8",
            )
            experiment = LoadedExperiment(config, run_dir, "hash", 10000)

            with patch(
                "tools.set_screening_status.load_experiment",
                return_value=experiment,
            ):
                set_status(config, "promoted", "top three at 20k")

            updated = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "promoted")
            self.assertEqual(updated["execution"], execution)
            self.assertEqual(
                updated["screening_history"][-1]["reason"],
                "top three at 20k",
            )


if __name__ == "__main__":
    unittest.main()
