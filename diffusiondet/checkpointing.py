"""Project checkpoint hooks with bounded weight retention."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path

from detectron2.engine.train_loop import HookBase


class LatestCheckpointer(HookBase):
    """Overwrite one resumable checkpoint periodically and at the final step."""

    def __init__(self, checkpointer, period: int, file_prefix: str = "model_latest"):
        if period <= 0:
            raise ValueError("checkpoint period must be positive")
        self._checkpointer = checkpointer
        self._period = int(period)
        self._file_prefix = file_prefix

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0 or next_iter >= self.trainer.max_iter:
            self._checkpointer.save(
                self._file_prefix,
                iteration=self.trainer.iter,
            )


class PersistentBestCheckpointer(HookBase):
    """Keep one best checkpoint and preserve its score across resume."""

    def __init__(
        self,
        eval_period: int,
        checkpointer,
        val_metric: str,
        mode: str = "max",
        file_prefix: str = "model_best",
        state_filename: str = "best_checkpoint.json",
    ):
        if eval_period <= 0:
            raise ValueError("evaluation period must be positive")
        if mode not in {"max", "min"}:
            raise ValueError("best checkpoint mode must be 'max' or 'min'")
        if not val_metric:
            raise ValueError("best checkpoint metric must not be empty")

        self._logger = logging.getLogger(__name__)
        self._period = int(eval_period)
        self._checkpointer = checkpointer
        self._val_metric = val_metric
        self._mode = mode
        self._file_prefix = file_prefix
        self._state_path = Path(checkpointer.save_dir) / state_filename
        self.best_metric = None
        self.best_iter = None

    def state_dict(self):
        if self.best_metric is None:
            return {}
        return {
            "metric_name": self._val_metric,
            "mode": self._mode,
            "best_metric": self.best_metric,
            "best_iter": self.best_iter,
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        if state_dict.get("metric_name") != self._val_metric:
            return
        if state_dict.get("mode") != self._mode:
            return
        self.best_metric = float(state_dict["best_metric"])
        self.best_iter = int(state_dict["best_iter"])

    def before_train(self):
        if not self._state_path.is_file():
            return
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.load_state_dict(state)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._logger.warning(
                "Ignoring invalid best-checkpoint state %s: %s",
                self._state_path,
                error,
            )

    def _is_better(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self._mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def _write_state(self):
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._state_path.name}.",
            suffix=".tmp",
            dir=self._state_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    self.state_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._state_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _best_checking(self):
        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metric_tuple is None:
            self._logger.warning(
                "Best-checkpoint metric %s was not computed; skipping save.",
                self._val_metric,
            )
            return

        latest_metric, metric_iter = metric_tuple
        latest_metric = float(latest_metric)
        metric_iter = int(metric_iter)
        if not math.isfinite(latest_metric):
            self._logger.warning(
                "Best-checkpoint metric %s is not finite: %r",
                self._val_metric,
                latest_metric,
            )
            return
        if not self._is_better(latest_metric):
            self._logger.info(
                "Keeping best %s=%0.5f @ iteration %s; latest is %0.5f.",
                self._val_metric,
                self.best_metric,
                self.best_iter,
                latest_metric,
            )
            return

        previous_checkpoint = self._checkpointer.get_checkpoint_file()
        self.best_metric = latest_metric
        self.best_iter = metric_iter
        self._checkpointer.save(
            self._file_prefix,
            iteration=metric_iter,
            best_metric=latest_metric,
            best_metric_name=self._val_metric,
        )
        if previous_checkpoint:
            self._checkpointer.tag_last_checkpoint(
                os.path.basename(previous_checkpoint)
            )
        self._write_state()
        self._logger.info(
            "Saved best checkpoint with %s=%0.5f @ iteration %s.",
            self._val_metric,
            self.best_metric,
            self.best_iter,
        )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if (
            next_iter % self._period == 0
            and next_iter < self.trainer.max_iter
        ):
            self._best_checking()

    def after_train(self):
        if self.trainer.iter >= self.trainer.max_iter:
            self._best_checking()
