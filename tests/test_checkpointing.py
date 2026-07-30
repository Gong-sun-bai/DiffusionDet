import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from diffusiondet.checkpointing import (
    LatestCheckpointer,
    PersistentBestCheckpointer,
)


class FakeCheckpointer:
    def __init__(self, save_dir):
        self.save_dir = str(save_dir)
        self.saved = []
        self.current_checkpoint = ""

    def save(self, name, **kwargs):
        self.saved.append((name, kwargs))
        self.current_checkpoint = str(Path(self.save_dir) / f"{name}.pth")
        Path(self.current_checkpoint).write_bytes(b"checkpoint")
        (Path(self.save_dir) / "last_checkpoint").write_text(
            f"{name}.pth",
            encoding="utf-8",
        )

    def get_checkpoint_file(self):
        return self.current_checkpoint

    def tag_last_checkpoint(self, basename):
        self.current_checkpoint = str(Path(self.save_dir) / basename)
        (Path(self.save_dir) / "last_checkpoint").write_text(
            basename,
            encoding="utf-8",
        )


class FakeStorage:
    def __init__(self, metrics=None):
        self.metrics = metrics or {}

    def latest(self):
        return self.metrics


class LatestCheckpointerTests(unittest.TestCase):
    def test_overwrites_one_latest_name_periodically_and_at_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpointer = FakeCheckpointer(temporary)
            hook = LatestCheckpointer(checkpointer, period=5)
            hook.trainer = SimpleNamespace(iter=4, max_iter=12)
            hook.after_step()
            hook.trainer.iter = 11
            hook.after_step()

            self.assertEqual(
                [name for name, _ in checkpointer.saved],
                ["model_latest", "model_latest"],
            )
            self.assertEqual(
                [value["iteration"] for _, value in checkpointer.saved],
                [4, 11],
            )
            self.assertEqual(
                [path.name for path in Path(temporary).glob("*.pth")],
                ["model_latest.pth"],
            )


class PersistentBestCheckpointerTests(unittest.TestCase):
    def test_keeps_best_score_across_resume_and_preserves_latest_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpointer = FakeCheckpointer(temporary)
            checkpointer.current_checkpoint = str(
                Path(temporary) / "model_latest.pth"
            )
            Path(checkpointer.current_checkpoint).write_bytes(b"checkpoint")
            storage = FakeStorage({"bbox/AP": (50.0, 4999)})
            trainer = SimpleNamespace(
                iter=4999,
                max_iter=10000,
                storage=storage,
            )

            hook = PersistentBestCheckpointer(
                5000,
                checkpointer,
                "bbox/AP",
            )
            hook.trainer = trainer
            hook.after_step()

            self.assertEqual(checkpointer.saved[0][0], "model_best")
            self.assertEqual(
                checkpointer.current_checkpoint,
                str(Path(temporary) / "model_latest.pth"),
            )
            self.assertEqual(
                (Path(temporary) / "last_checkpoint").read_text(),
                "model_latest.pth",
            )
            self.assertEqual(
                sorted(path.name for path in Path(temporary).glob("*.pth")),
                ["model_best.pth", "model_latest.pth"],
            )
            state = json.loads(
                (Path(temporary) / "best_checkpoint.json").read_text()
            )
            self.assertEqual(state["best_metric"], 50.0)
            self.assertEqual(state["best_iter"], 4999)

            resumed_hook = PersistentBestCheckpointer(
                5000,
                checkpointer,
                "bbox/AP",
            )
            resumed_hook.trainer = trainer
            resumed_hook.before_train()
            self.assertEqual(resumed_hook.best_metric, 50.0)

            storage.metrics["bbox/AP"] = (49.0, 9999)
            trainer.iter = 9999
            resumed_hook.after_train()
            self.assertEqual(len(checkpointer.saved), 1)


if __name__ == "__main__":
    unittest.main()
