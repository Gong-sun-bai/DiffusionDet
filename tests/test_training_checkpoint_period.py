import unittest

from train_net import Trainer


class ScreeningCheckpointPeriodTests(unittest.TestCase):
    def test_bounded_screening_run_uses_dense_recovery_checkpoint(self):
        self.assertEqual(Trainer.checkpoint_period(5000, 50000, 254264), 1000)

    def test_shorter_configured_period_is_preserved(self):
        self.assertEqual(Trainer.checkpoint_period(200, 5000, 254264), 200)

    def test_full_training_keeps_configured_period(self):
        self.assertEqual(Trainer.checkpoint_period(5000, None, 254264), 5000)
        self.assertEqual(Trainer.checkpoint_period(5000, 254264, 254264), 5000)

    def test_recovered_rung_runs_hooks_at_exact_iteration_without_step(self):
        trainer = object.__new__(Trainer)
        calls = []
        trainer.before_train = lambda: calls.append(
            ("before", trainer.storage.iter)
        )
        trainer.after_train = lambda: calls.append(
            ("after", trainer.storage.iter)
        )
        trainer._last_eval_results = {"bbox": {"AP": 12.3}}

        result = trainer.evaluate_resumed_rung(5000)

        self.assertEqual(result, {"bbox": {"AP": 12.3}})
        self.assertEqual(calls, [("before", 5000), ("after", 5000)])
        self.assertEqual(trainer.iter, 5000)
        self.assertEqual(trainer.start_iter, 5000)
        self.assertEqual(trainer.max_iter, 5000)


if __name__ == "__main__":
    unittest.main()
