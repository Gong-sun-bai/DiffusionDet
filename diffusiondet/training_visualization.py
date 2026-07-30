"""YOLO-style training result plots backed by Detectron2 ``metrics.json``."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from detectron2.engine.train_loop import HookBase
from detectron2.utils.events import EventWriter, get_event_storage, has_event_storage


LOGGER = logging.getLogger(__name__)

TRAIN_METRICS = (
    "total_loss",
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "lr",
    "time",
    "data_time",
)
VALIDATION_METRICS = (
    "bbox/AP",
    "bbox/AP50",
    "bbox/AP75",
    "bbox/APs",
    "bbox/APm",
    "bbox/APl",
)
SUPPORTED_METRICS = frozenset(TRAIN_METRICS + VALIDATION_METRICS)


class TrainingVisualizationError(RuntimeError):
    """Raised when metrics cannot be parsed or rendered safely."""


@dataclass(frozen=True)
class MetricsHistory:
    """Merged, iteration-sorted metric records."""

    rows: tuple[dict[str, Any], ...]
    training_records: int
    validation_records: int

    @property
    def latest_iteration(self) -> int:
        return int(self.rows[-1]["iteration"])

    def series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        points = [
            (int(row["iteration"]), float(row[name]))
            for row in self.rows
            if name in row and _is_finite_number(row[name])
        ]
        if not points:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
        x_values, y_values = zip(*points)
        return (
            np.asarray(x_values, dtype=np.int64),
            np.asarray(y_values, dtype=np.float64),
        )


@dataclass(frozen=True)
class PlotSummary:
    """Summary returned to the CLI and tests after a successful render."""

    output_path: Path
    training_records: int
    validation_records: int
    latest_iteration: int


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _normalise_iteration(value: Any, *, line_number: int | None = None) -> int:
    location = f"第 {line_number} 行" if line_number is not None else "附加指标"
    if not _is_finite_number(value) or not float(value).is_integer():
        raise TrainingVisualizationError(f"{location}的 iteration 必须是有限整数。")
    iteration = int(value)
    if iteration < 0:
        raise TrainingVisualizationError(f"{location}的 iteration 不能为负数。")
    return iteration


def _merge_metric_row(
    merged: dict[int, dict[str, Any]],
    row: Any,
    *,
    line_number: int | None = None,
) -> None:
    location = f"第 {line_number} 行" if line_number is not None else "附加指标"
    if not isinstance(row, dict):
        raise TrainingVisualizationError(f"{location}必须是 JSON 对象。")
    if "iteration" not in row:
        raise TrainingVisualizationError(f"{location}缺少 iteration。")
    iteration = _normalise_iteration(row["iteration"], line_number=line_number)
    target = merged.setdefault(iteration, {"iteration": iteration})
    target.update(row)
    target["iteration"] = iteration


def load_metrics_history(
    metrics_path: str | Path,
    *,
    extra_rows: Iterable[Mapping[str, Any]] = (),
) -> MetricsHistory:
    """Load JSON Lines metrics and merge records sharing an iteration.

    A malformed final non-empty line is ignored with a warning because a live
    writer may have been observed between writing and flushing that line.
    Malformed records anywhere else are treated as evidence corruption.
    """

    path = Path(metrics_path)
    if not path.is_file():
        raise TrainingVisualizationError(f"指标文件不存在：{path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TrainingVisualizationError(f"无法读取指标文件 {path}：{error}") from error

    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty_indices:
        raise TrainingVisualizationError(f"指标文件为空：{path}")
    final_nonempty_index = nonempty_indices[-1]

    merged: dict[int, dict[str, Any]] = {}
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        line_number = index + 1
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            if index == final_nonempty_index:
                warnings.warn(
                    f"忽略 metrics.json 最后一条未完成记录（第 {line_number} 行）："
                    f"{error.msg}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise TrainingVisualizationError(
                f"metrics.json 第 {line_number} 行不是合法 JSON：{error.msg}"
            ) from error
        _merge_metric_row(merged, row, line_number=line_number)

    for row in extra_rows:
        _merge_metric_row(merged, dict(row))

    rows = tuple(merged[iteration] for iteration in sorted(merged))
    if not rows:
        raise TrainingVisualizationError(f"指标文件没有可解析记录：{path}")

    has_drawable_metric = any(
        _is_finite_number(row.get(name))
        for row in rows
        for name in SUPPORTED_METRICS
    )
    if not has_drawable_metric:
        raise TrainingVisualizationError(
            f"指标文件没有可绘制的训练或 bbox 验证指标：{path}"
        )

    training_records = sum(
        any(_is_finite_number(row.get(name)) for name in TRAIN_METRICS)
        for row in rows
    )
    validation_records = sum(
        any(_is_finite_number(row.get(name)) for name in VALIDATION_METRICS)
        for row in rows
    )
    return MetricsHistory(rows, training_records, validation_records)


def read_max_iter(config_path: str | Path) -> int | None:
    """Read ``SOLVER.MAX_ITER`` from a frozen config when available."""

    path = Path(config_path)
    if not path.is_file():
        return None
    try:
        import yaml
    except ImportError:
        LOGGER.warning("PyYAML 不可用，将按现有迭代绘图。")
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        value = config["SOLVER"]["MAX_ITER"]
        return int(value) if int(value) > 0 else None
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        LOGGER.warning("无法从 %s 读取 SOLVER.MAX_ITER，将按现有迭代绘图。", path)
        return None


def _plot_series(
    axis,
    history: MetricsHistory,
    series: Sequence[tuple[str, str]],
    *,
    title: str,
    ylabel: str,
    empty_message: str,
    log_scale_if_positive: bool = False,
) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    plotted = []
    all_positive = True
    for index, (metric_name, label) in enumerate(series):
        x_values, y_values = history.series(metric_name)
        if not len(x_values):
            continue
        plotted.append(label)
        all_positive = all_positive and bool(np.all(y_values > 0))
        if len(x_values) == 1:
            axis.scatter(x_values, y_values, s=42, label=label, zorder=3)
            axis.annotate(
                f"{y_values[0]:.4g}",
                (x_values[0], y_values[0]),
                xytext=(6, 6 + index * 3),
                textcoords="offset points",
                fontsize=8,
            )
        else:
            axis.plot(x_values, y_values, linewidth=1.5, label=label)

    axis.set_title(title)
    axis.set_xlabel("Iteration")
    axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{int(round(value)):,}")
    )
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(fontsize=8)
        if log_scale_if_positive and all_positive:
            axis.set_yscale("log")
    else:
        axis.text(
            0.5,
            0.5,
            empty_message,
            ha="center",
            va="center",
            transform=axis.transAxes,
            color="0.45",
        )


def render_results_plot(
    history: MetricsHistory,
    output_path: str | Path,
    *,
    run_name: str,
    max_iter: int | None = None,
) -> Path:
    """Render a fixed 2x3 results dashboard and atomically replace the PNG."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_path)
    if not output.parent.is_dir():
        raise TrainingVisualizationError(f"输出目录不存在：{output.parent}")

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    _plot_series(
        axes[0, 0],
        history,
        (("total_loss", "total_loss"),),
        title="Total Loss",
        ylabel="Loss",
        empty_message="No total loss recorded",
    )
    _plot_series(
        axes[0, 1],
        history,
        (
            ("loss_ce", "classification"),
            ("loss_bbox", "bbox L1"),
            ("loss_giou", "GIoU"),
        ),
        title="Main-head Losses",
        ylabel="Loss",
        empty_message="No component losses recorded",
    )
    _plot_series(
        axes[0, 2],
        history,
        (("lr", "learning rate"),),
        title="Learning Rate",
        ylabel="LR",
        empty_message="No learning rate recorded",
        log_scale_if_positive=True,
    )
    _plot_series(
        axes[1, 0],
        history,
        (("time", "iteration"), ("data_time", "data loading")),
        title="Iteration Time",
        ylabel="Seconds",
        empty_message="No timing metrics recorded",
    )
    _plot_series(
        axes[1, 1],
        history,
        (
            ("bbox/AP", "AP"),
            ("bbox/AP50", "AP50"),
            ("bbox/AP75", "AP75"),
        ),
        title="Validation AP by IoU",
        ylabel="AP",
        empty_message="No validation metrics yet",
    )
    _plot_series(
        axes[1, 2],
        history,
        (
            ("bbox/APs", "small"),
            ("bbox/APm", "medium"),
            ("bbox/APl", "large"),
        ),
        title="Validation AP by Size",
        ylabel="AP",
        empty_message="No validation metrics yet",
    )

    progress = f"iteration {history.latest_iteration}"
    if max_iter is not None:
        progress += f" / {max_iter}"
    figure.suptitle(f"{run_name}\n{progress}", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.stem}.",
            suffix=".tmp.png",
            dir=output.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        figure.savefig(temporary_path, dpi=150, format="png")
        os.replace(temporary_path, output)
        temporary_path = None
    except OSError as error:
        raise TrainingVisualizationError(f"无法原子写入 {output}：{error}") from error
    finally:
        plt.close(figure)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output


def generate_results_plot(
    run_dir: str | Path,
    *,
    max_iter: int | None = None,
    extra_rows: Iterable[Mapping[str, Any]] = (),
) -> PlotSummary:
    """Read a run directory and generate its canonical ``results.png``."""

    directory = Path(run_dir)
    if not directory.is_dir():
        raise TrainingVisualizationError(f"结果目录不存在：{directory}")
    if max_iter is None:
        max_iter = read_max_iter(directory / "config.yaml")
    history = load_metrics_history(
        directory / "metrics.json",
        extra_rows=extra_rows,
    )
    output = render_results_plot(
        history,
        directory / "results.png",
        run_name=directory.name,
        max_iter=max_iter,
    )
    return PlotSummary(
        output_path=output,
        training_records=history.training_records,
        validation_records=history.validation_records,
        latest_iteration=history.latest_iteration,
    )


def _latest_storage_rows() -> tuple[dict[str, Any], ...]:
    if not has_event_storage():
        return ()
    grouped: dict[int, dict[str, Any]] = {}
    for name, (value, iteration) in get_event_storage().latest().items():
        if not _is_finite_number(iteration):
            continue
        row = grouped.setdefault(int(iteration), {"iteration": int(iteration)})
        row[name] = value
    return tuple(grouped[iteration] for iteration in sorted(grouped))


class TrainingPlotWriter(EventWriter):
    """Non-fatal writer that renders the latest training dashboard."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_iter: int | None = None,
        render: Callable[..., PlotSummary] = generate_results_plot,
    ):
        self._output_dir = Path(output_dir)
        self._max_iter = max_iter
        self._render = render
        self._last_error: str | None = None

    def write(self) -> None:
        try:
            self._render(
                self._output_dir,
                max_iter=self._max_iter,
                extra_rows=_latest_storage_rows(),
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if message != self._last_error:
                LOGGER.warning(
                    "训练结果图更新失败；训练将继续。%s",
                    message,
                    exc_info=True,
                )
            self._last_error = message
        else:
            if self._last_error is not None:
                LOGGER.info("训练结果图已恢复正常更新。")
            self._last_error = None


class TrainingPlotHook(HookBase):
    """Refresh plots periodically, after validation, and after training."""

    def __init__(self, writer: EventWriter, *, period: int, eval_period: int = 0):
        if period <= 0:
            raise ValueError("TRAINING_PLOTS.PERIOD 必须大于 0。")
        if eval_period < 0:
            raise ValueError("TEST.EVAL_PERIOD 不能为负数。")
        self._writer = writer
        self._period = int(period)
        self._eval_period = int(eval_period)

    def after_step(self) -> None:
        next_iteration = self.trainer.iter + 1
        periodic = next_iteration % self._period == 0
        after_evaluation = (
            self._eval_period > 0
            and next_iteration % self._eval_period == 0
            and next_iteration != self.trainer.max_iter
        )
        if periodic or after_evaluation:
            self._writer.write()

    def after_train(self) -> None:
        self._writer.write()
        self._writer.close()
