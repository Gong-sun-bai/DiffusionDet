import tempfile
import unittest
from pathlib import Path

from tools.migrate_historical_runs import raw_snapshot


class RawSnapshotTests(unittest.TestCase):
    def test_timestamped_evaluations_are_not_historical_raw_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "config.yaml").write_text("MODEL: {}\n", encoding="utf-8")
            (run_dir / "log.txt").write_text("historical\n", encoding="utf-8")
            (run_dir / "metrics.json").write_text(
                '{"bbox/AP": 1.0, "iteration": 1}\n', encoding="utf-8"
            )
            (run_dir / "last_checkpoint").write_text(
                "model_final.pth\n", encoding="utf-8"
            )
            (run_dir / "model_final.pth").write_bytes(b"weights")
            evaluation = run_dir / "evaluations" / "20260805T010203.000000Z"
            evaluation.mkdir(parents=True)
            (evaluation / "metrics.json").write_text(
                '{"bbox/AP": 99.0}\n', encoding="utf-8"
            )

            snapshot = raw_snapshot(run_dir)

            self.assertEqual(snapshot["file_count"], 5)
            self.assertEqual(snapshot["final_metrics"]["AP"], 1.0)


if __name__ == "__main__":
    unittest.main()
