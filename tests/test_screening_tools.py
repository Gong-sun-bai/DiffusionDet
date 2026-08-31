import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.evaluate_head_stages import newest_summary
from tools.evaluate_sampling_variants import DEFAULT_VARIANTS, parse_variant
from tools.summarize_screening import compare_rows, select_metric_row


class HeadStageSummaryTests(unittest.TestCase):
    def write_evaluation(self, root, name, *, stage, seed):
        directory = root / name
        directory.mkdir()
        config = {
            "SEED": seed,
            "MODEL": {
                "DiffusionDet": {
                    "INFERENCE_HEAD_STAGE": stage,
                    "SAMPLE_STEP": 1,
                    "ENSEMBLE_MODE": "legacy_nonfinal",
                }
            },
        }
        (directory / "config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        (directory / "result_summary.json").write_text(
            json.dumps({"results": {"bbox": {"AP": stage}}}),
            encoding="utf-8",
        )
        return directory

    def test_ignores_concurrent_evaluation_with_different_seed_or_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_evaluation(root, "wrong-stage", stage=6, seed=40244023)
            expected = self.write_evaluation(
                root, "expected", stage=4, seed=40244023
            )
            self.write_evaluation(root, "wrong-seed", stage=4, seed=-1)

            directory, results = newest_summary(
                root,
                set(),
                stage=4,
                sample_step=1,
                ensemble_mode="legacy_nonfinal",
                seed=40244023,
            )

            self.assertEqual(directory, expected)
            self.assertEqual(results["bbox"]["AP"], 4)


class RankingTests(unittest.TestCase):
    def test_exact_rung_does_not_leak_best_metric_from_another_rung(self):
        rows = [
            {"iteration": 200, "bbox/AP": 5.0},
            {"iteration": 5000, "bbox/AP": 4.0},
        ]
        self.assertEqual(select_metric_row(rows, 5000)["bbox/AP"], 4.0)
        self.assertEqual(select_metric_row(rows, None)["bbox/AP"], 5.0)

    def test_missing_exact_rung_is_explicit(self):
        self.assertEqual(select_metric_row([{"iteration": 200}], 5000), {})

    def test_ap_difference_over_half_wins(self):
        stronger_ap = {"AP": 60.6, "AP75": 50.0, "APs": 50.0}
        stronger_ap75 = {"AP": 60.0, "AP75": 90.0, "APs": 90.0}
        self.assertLess(compare_rows(stronger_ap, stronger_ap75), 0)

    def test_ap75_breaks_tie_within_half_ap(self):
        stronger_ap = {"AP": 60.4, "AP75": 50.0, "APs": 50.0}
        stronger_ap75 = {"AP": 60.0, "AP75": 51.0, "APs": 49.0}
        self.assertGreater(compare_rows(stronger_ap, stronger_ap75), 0)


class SamplingMatrixTests(unittest.TestCase):
    def test_default_matrix_covers_all_step_mode_pairs(self):
        self.assertEqual(len(DEFAULT_VARIANTS), 9)
        self.assertEqual(
            set(DEFAULT_VARIANTS),
            {
                (steps, mode)
                for steps in (1, 2, 4)
                for mode in ("legacy_nonfinal", "final_only", "all_steps")
            },
        )

    def test_variant_parser(self):
        self.assertEqual(parse_variant("4:all_steps"), (4, "all_steps"))
        with self.assertRaises(Exception):
            parse_variant("3:all_steps")


if __name__ == "__main__":
    unittest.main()
