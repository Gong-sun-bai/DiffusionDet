import tempfile
import unittest
from pathlib import Path

from tools.run_under10m_remaining_screening import (
    evaluate_distillation_gates,
    eta_hours_for_three,
    is_resource_failure,
    parse_peak_memory_mib,
    render_distillation_config,
    select_promoted_ids,
)


class PromotionTests(unittest.TestCase):
    def test_top_three_plus_distinct_ap75_leader(self):
        rows = [
            {"id": "a", "AP75": 50.0},
            {"id": "b", "AP75": 49.0},
            {"id": "c", "AP75": 48.0},
            {"id": "d", "AP75": 80.0},
            {"id": "e", "AP75": 20.0},
        ]
        self.assertEqual(select_promoted_ids(rows), ["a", "b", "c", "d"])

    def test_no_duplicate_when_ap75_leader_is_already_top_three(self):
        rows = [
            {"id": "a", "AP75": 80.0},
            {"id": "b", "AP75": 49.0},
            {"id": "c", "AP75": 48.0},
            {"id": "d", "AP75": 50.0},
        ]
        self.assertEqual(select_promoted_ids(rows), ["a", "b", "c"])


class ResourceTests(unittest.TestCase):
    def test_parses_largest_peak_memory(self):
        self.assertEqual(
            parse_peak_memory_mib("max_mem: 100M\nmax_mem: 20479M\n"), 20479
        )
        self.assertIsNone(parse_peak_memory_mib("no memory evidence"))

    def test_only_memory_failures_are_expected_resource_rejections(self):
        self.assertTrue(is_resource_failure("RuntimeError: CUDA out of memory"))
        self.assertFalse(is_resource_failure("No CUDA GPUs are available"))
        self.assertFalse(is_resource_failure("GPU has fallen off the bus"))


class DistillationGateTests(unittest.TestCase):
    def rows(self, ap=60.0, ap75=70.0):
        return {"AP": ap, "AP75": ap75}

    def test_all_gates_pass(self):
        resource = {"passed": True}
        result = evaluate_distillation_gates(
            self.rows(), self.rows(61.0, 70.0), resource, 1.0, 1.2
        )
        self.assertTrue(result["eligible_to_replace_second_finalist"])

    def test_ap_drop_and_eta_are_hard_gates(self):
        resource = {"passed": True}
        result = evaluate_distillation_gates(
            self.rows(), self.rows(59.6, 72.0), resource, 1.0, 2.0
        )
        self.assertFalse(result["AP_floor_passed"])
        self.assertFalse(result["eta_passed"])
        self.assertFalse(result["eligible_to_replace_second_finalist"])

    def test_eta_includes_two_runs_and_slower_winner_repeat(self):
        expected = (1.0 + 2.0 + 2.0) * 254264 / 3600.0 + 12.0
        self.assertAlmostEqual(eta_hours_for_three(1.0, 2.0), expected)


class ConfigGenerationTests(unittest.TestCase):
    def test_generated_config_inherits_selected_student_and_enables_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "sar-014.yaml"
            text = render_distillation_config(
                {
                    "id": "sar-013",
                    "config": "configs/experiments/sar_ship/sar-013-under10m-repvit-pretrained.yaml",
                },
                destination,
            )
        self.assertIn("sar-013-under10m-repvit-pretrained.yaml", text)
        self.assertIn('ID: "sar-014"', text)
        self.assertIn("ENABLED: True", text)
        self.assertIn("model_best.pth", text)


if __name__ == "__main__":
    unittest.main()
