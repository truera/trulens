"""Alignment diagnostics for benchmark and judge-alignment results."""

from __future__ import annotations

import collections.abc as collections_abc
import html
import numbers
import typing
import warnings

import numpy as np
import pandas as pd
from trulens.feedback import groundtruth as feedback_groundtruth

_INVALID_INDEX_LIMIT = 5
_T = typing.TypeVar("_T")


def _is_sequence_like(value: typing.Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, collections_abc.Mapping)):
        return False
    if isinstance(value, (np.ndarray, pd.Series)):
        return True
    return isinstance(value, collections_abc.Sequence)


def _is_pair_like(value: typing.Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, collections_abc.Mapping)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size > 0
    return isinstance(value, collections_abc.Sequence) and len(value) > 0


def _first_pair_value(value: typing.Any) -> typing.Any:
    if isinstance(value, np.ndarray):
        return value.flat[0]
    return value[0]


def _format_indexes(indexes: typing.Sequence[int]) -> str:
    shown = ", ".join(
        str(int(index)) for index in indexes[:_INVALID_INDEX_LIMIT]
    )
    if len(indexes) > _INVALID_INDEX_LIMIT:
        shown = f"{shown}, ..."
    return shown


def _extract_benchmark_scores(predicted_scores: typing.Any) -> typing.Any:
    """Extract scores from ``(scores, meta_scores)`` benchmark outputs."""

    if not isinstance(predicted_scores, tuple) or len(predicted_scores) != 2:
        return predicted_scores

    scores, _meta_scores = predicted_scores
    if _is_sequence_like(scores):
        return scores

    return predicted_scores


def _values_from_vector(
    values: typing.Any, *, name: str, extract_pairs: bool
) -> list[typing.Any]:
    if isinstance(values, pd.Series):
        return values.tolist()

    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return [values.item()]
        if values.ndim == 1:
            return values.tolist()
        if extract_pairs and values.ndim == 2 and values.shape[1] > 0:
            return values[:, 0].tolist()
        raise ValueError(f"{name} must be a one-dimensional sequence.")

    if not _is_sequence_like(values):
        raise ValueError(f"{name} must be a list-like sequence of values.")

    return list(values)


def _coerce_numeric_vector(
    values: typing.Any, *, name: str, extract_pairs: bool = False
) -> np.ndarray:
    if extract_pairs:
        values = _extract_benchmark_scores(values)

    raw_values = _values_from_vector(
        values, name=name, extract_pairs=extract_pairs
    )

    if (
        extract_pairs
        and raw_values
        and all(_is_pair_like(value) for value in raw_values)
    ):
        raw_values = [_first_pair_value(value) for value in raw_values]

    if not raw_values:
        raise ValueError(f"{name} must contain at least one value.")

    converted: list[float] = []
    invalid_indexes: list[int] = []
    for index, value in enumerate(raw_values):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            invalid_indexes.append(index)
            continue

        if not np.isfinite(numeric_value):
            invalid_indexes.append(index)
            continue

        converted.append(numeric_value)

    if invalid_indexes:
        indexes = _format_indexes(invalid_indexes)
        raise ValueError(
            f"{name} must contain only finite numeric values; invalid "
            f"value(s) at index(es): {indexes}."
        )

    array = np.asarray(converted, dtype=float)
    out_of_range = np.flatnonzero((array < 0.0) | (array > 1.0))
    if len(out_of_range) > 0:
        indexes = _format_indexes(out_of_range.tolist())
        raise ValueError(
            f"{name} values must be in the [0.0, 1.0] range; out-of-range "
            f"value(s) at index(es): {indexes}."
        )

    return array


def _coerce_threshold(value: float, *, name: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value.") from exc

    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"{name} must be in the [0.0, 1.0] range.")

    return threshold


def _coerce_thresholds(
    thresholds: typing.Sequence[float] | None, *, threshold: float
) -> list[float]:
    if thresholds is None:
        return [threshold]

    if isinstance(thresholds, (pd.Series, np.ndarray)):
        raw_thresholds = thresholds.tolist()
    else:
        try:
            raw_thresholds = list(thresholds)
        except TypeError as exc:
            raise ValueError(
                "thresholds must be a sequence of finite numeric values."
            ) from exc

    if not raw_thresholds:
        raise ValueError("thresholds must contain at least one value.")

    return [
        _coerce_threshold(value, name="thresholds") for value in raw_thresholds
    ]


def _coerce_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a positive integer.")

    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

    return value


def _coerce_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a non-negative integer.")

    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")

    return value


def _metric_float(value: typing.Any) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if np.isinf(metric):
        return float("nan")

    return metric


def _pairwise(
    values: collections_abc.Iterable[_T],
) -> collections_abc.Iterator[tuple[_T, _T]]:
    iterator = iter(values)
    try:
        previous = next(iterator)
    except StopIteration:
        return

    for current in iterator:
        yield previous, current
        previous = current


class AlignmentReport:
    """Generate alignment diagnostics for benchmark predictions.

    Args:
        predicted_scores: Predicted judge scores. This may be a one-dimensional
            sequence, a pandas series, a numpy array, score-confidence pairs, or
            a ``TruBenchmarkExperiment`` result of ``(scores, meta_scores)``.
        true_labels: Ground-truth labels or scores in the same order as the
            predictions.
        examples: Optional examples used only when displaying worst misses.
        threshold: Threshold used for binary alignment metrics.
        thresholds: Optional thresholds for confusion-matrix rows.
        n_bins: Number of bins for calibration and distribution data.
        top_n: Number of worst misses to include.
    """

    def __init__(
        self,
        predicted_scores: typing.Any,
        true_labels: typing.Any,
        examples: typing.Any | None = None,
        threshold: float = 0.5,
        thresholds: typing.Sequence[float] | None = None,
        n_bins: int = 10,
        top_n: int = 10,
    ) -> None:
        self.predicted_scores = _coerce_numeric_vector(
            predicted_scores, name="predicted_scores", extract_pairs=True
        )
        self.true_labels = _coerce_numeric_vector(
            true_labels, name="true_labels"
        )

        if len(self.predicted_scores) != len(self.true_labels):
            raise ValueError(
                "predicted_scores and true_labels must have equal length; "
                f"got {len(self.predicted_scores)} and "
                f"{len(self.true_labels)}."
            )

        self.examples = examples
        self.threshold = _coerce_threshold(threshold, name="threshold")
        self.thresholds = _coerce_thresholds(
            thresholds, threshold=self.threshold
        )
        self.n_bins = _coerce_positive_int(n_bins, name="n_bins")
        self.top_n = _coerce_nonnegative_int(top_n, name="top_n")

    def _new_aggregator(
        self, labels: np.ndarray | None = None
    ) -> feedback_groundtruth.GroundTruthAggregator:
        labels = self.true_labels if labels is None else labels
        return feedback_groundtruth.GroundTruthAggregator(
            true_labels=np.asarray(labels, dtype=float).tolist()
        )

    def _metric(
        self,
        name: str,
        *,
        labels: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        **kwargs: typing.Any,
    ) -> float:
        aggregator = self._new_aggregator(labels)
        metric = getattr(aggregator, name)
        metric_scores = self.predicted_scores if scores is None else scores

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = metric(
                    np.asarray(metric_scores, dtype=float).tolist(), **kwargs
                )
        except (AssertionError, TypeError, ValueError, ZeroDivisionError):
            return float("nan")

        return _metric_float(value)

    def _binary_labels(self, threshold: float) -> np.ndarray:
        return (self.true_labels >= threshold).astype(int)

    def _binary_predictions(self, threshold: float) -> np.ndarray:
        return (self.predicted_scores >= threshold).astype(int)

    def _auc(self) -> float:
        binary_labels = self._binary_labels(self.threshold)
        if len(np.unique(binary_labels)) < 2:
            return float("nan")

        return self._metric("auc", labels=binary_labels)

    def _summary_df(self) -> pd.DataFrame:
        rows = [
            {"metric": "MAE", "value": self._metric("mae")},
            {
                "metric": "Spearman correlation",
                "value": self._metric("spearman_correlation"),
            },
            {
                "metric": "Kendall's tau",
                "value": self._metric("kendall_tau"),
            },
            {
                "metric": f"Cohen's kappa at {self.threshold:g}",
                "value": self._metric("cohens_kappa", threshold=self.threshold),
            },
            {
                "metric": "Brier score",
                "value": self._metric("brier_score"),
            },
            {"metric": "AUC", "value": self._auc()},
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def _confusion_matrix_df(self) -> pd.DataFrame:
        rows = []
        for threshold in self.thresholds:
            true_binary = self._binary_labels(threshold)
            predicted_binary = self._binary_predictions(threshold)

            rows.append({
                "threshold": threshold,
                "TN": int(np.sum((true_binary == 0) & (predicted_binary == 0))),
                "FP": int(np.sum((true_binary == 0) & (predicted_binary == 1))),
                "FN": int(np.sum((true_binary == 1) & (predicted_binary == 0))),
                "TP": int(np.sum((true_binary == 1) & (predicted_binary == 1))),
            })

        return pd.DataFrame(rows, columns=["threshold", "TN", "FP", "FN", "TP"])

    def _bin_mask(
        self, values: np.ndarray, lower: float, upper: float, is_last: bool
    ) -> np.ndarray:
        if is_last:
            return (values >= lower) & (values <= upper)
        return (values >= lower) & (values < upper)

    def _calibration_df(self) -> pd.DataFrame:
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        rows = []

        for bin_index, (lower, upper) in enumerate(_pairwise(bin_edges)):
            in_bin = self._bin_mask(
                self.predicted_scores,
                lower,
                upper,
                bin_index == self.n_bins - 1,
            )
            count = int(np.sum(in_bin))
            rows.append({
                "bin": bin_index,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "mean_predicted_score": (
                    float(np.mean(self.predicted_scores[in_bin]))
                    if count
                    else float("nan")
                ),
                "mean_true_label": (
                    float(np.mean(self.true_labels[in_bin]))
                    if count
                    else float("nan")
                ),
            })

        return pd.DataFrame(
            rows,
            columns=[
                "bin",
                "bin_lower",
                "bin_upper",
                "count",
                "mean_predicted_score",
                "mean_true_label",
            ],
        )

    def _score_distribution_df(self) -> pd.DataFrame:
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        rows = []

        for bin_index, (lower, upper) in enumerate(_pairwise(bin_edges)):
            is_last = bin_index == self.n_bins - 1
            predicted_mask = self._bin_mask(
                self.predicted_scores, lower, upper, is_last
            )
            true_mask = self._bin_mask(self.true_labels, lower, upper, is_last)
            rows.append({
                "bin": bin_index,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "predicted_count": int(np.sum(predicted_mask)),
                "true_label_count": int(np.sum(true_mask)),
            })

        return pd.DataFrame(
            rows,
            columns=[
                "bin",
                "bin_lower",
                "bin_upper",
                "predicted_count",
                "true_label_count",
            ],
        )

    def _examples_df(self) -> pd.DataFrame | None:
        if self.examples is None:
            return None

        if isinstance(self.examples, pd.DataFrame):
            return self.examples.reset_index(drop=True)

        if isinstance(self.examples, pd.Series):
            return pd.DataFrame({"example": self.examples.tolist()})

        if isinstance(self.examples, collections_abc.Mapping):
            return pd.DataFrame([self.examples])

        try:
            examples = list(self.examples)
        except TypeError:
            return pd.DataFrame({"example": [self.examples]})

        if examples and all(
            isinstance(example, collections_abc.Mapping) for example in examples
        ):
            return pd.DataFrame(examples)

        return pd.DataFrame({"example": examples})

    def _worst_misses_df(self) -> pd.DataFrame:
        base_df = pd.DataFrame({
            "index": np.arange(len(self.predicted_scores)),
            "predicted_score": self.predicted_scores,
            "true_label": self.true_labels,
            "absolute_error": np.abs(self.predicted_scores - self.true_labels),
        })

        examples_df = self._examples_df()
        if examples_df is not None:
            examples_df = examples_df.reindex(
                range(len(self.predicted_scores))
            ).reset_index(drop=True)
            examples_df = examples_df.rename(
                columns={
                    column: f"example_{column}"
                    for column in examples_df.columns
                    if column in base_df.columns
                }
            )
            base_df = pd.concat([base_df, examples_df], axis=1)

        return (
            base_df
            .sort_values(
                ["absolute_error", "index"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(self.top_n)
            .reset_index(drop=True)
        )

    def _difficulty_breakdown_df(self) -> pd.DataFrame:
        buckets = [
            ("easy", 0.0, 0.3, False, "[0.0, 0.3)"),
            ("medium", 0.3, 0.7, False, "[0.3, 0.7)"),
            ("hard", 0.7, 1.0, True, "[0.7, 1.0]"),
        ]
        rows = []

        for bucket, lower, upper, include_upper, label in buckets:
            if include_upper:
                in_bucket = (self.true_labels >= lower) & (
                    self.true_labels <= upper
                )
            else:
                in_bucket = (self.true_labels >= lower) & (
                    self.true_labels < upper
                )

            count = int(np.sum(in_bucket))
            if count == 0:
                rows.append({
                    "bucket": bucket,
                    "range": label,
                    "count": count,
                    "mae": float("nan"),
                    "mean_predicted_score": float("nan"),
                    "mean_true_label": float("nan"),
                    "spearman_correlation": float("nan"),
                })
                continue

            bucket_scores = self.predicted_scores[in_bucket]
            bucket_labels = self.true_labels[in_bucket]
            spearman = (
                self._metric(
                    "spearman_correlation",
                    labels=bucket_labels,
                    scores=bucket_scores,
                )
                if count > 1
                else float("nan")
            )
            rows.append({
                "bucket": bucket,
                "range": label,
                "count": count,
                "mae": self._metric(
                    "mae", labels=bucket_labels, scores=bucket_scores
                ),
                "mean_predicted_score": float(np.mean(bucket_scores)),
                "mean_true_label": float(np.mean(bucket_labels)),
                "spearman_correlation": spearman,
            })

        return pd.DataFrame(
            rows,
            columns=[
                "bucket",
                "range",
                "count",
                "mae",
                "mean_predicted_score",
                "mean_true_label",
                "spearman_correlation",
            ],
        )

    def to_dataframe(self) -> dict[str, pd.DataFrame]:
        """Return report sections as pandas dataframes."""

        return {
            "summary": self._summary_df(),
            "confusion_matrix": self._confusion_matrix_df(),
            "calibration": self._calibration_df(),
            "score_distribution": self._score_distribution_df(),
            "worst_misses": self._worst_misses_df(),
            "difficulty_breakdown": self._difficulty_breakdown_df(),
        }

    def print_summary(self) -> str:
        """Print and return a concise console-friendly report."""

        frames = self.to_dataframe()
        with pd.option_context(
            "display.max_columns",
            8,
            "display.max_colwidth",
            60,
            "display.width",
            120,
        ):
            lines = [
                "Alignment Report",
                f"Examples: {len(self.predicted_scores)}",
                f"Threshold: {self.threshold:g}",
                "",
                "Summary metrics",
                frames["summary"].to_string(index=False),
                "",
                "Confusion matrix",
                frames["confusion_matrix"].to_string(index=False),
                "",
                "Difficulty breakdown",
                frames["difficulty_breakdown"].to_string(index=False),
                "",
                "Worst misses",
                frames["worst_misses"].to_string(index=False),
            ]

        report = "\n".join(lines)
        print(report)
        return report

    def plot(self) -> dict[str, typing.Any]:
        """Return matplotlib figures for calibration and score distribution."""

        import matplotlib.pyplot as plt

        figures = {}
        calibration_df = self._calibration_df()
        non_empty_bins = calibration_df[calibration_df["count"] > 0]

        calibration_fig, calibration_ax = plt.subplots()
        calibration_ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            color="gray",
            label="Perfect alignment",
        )
        if not non_empty_bins.empty:
            calibration_ax.plot(
                non_empty_bins["mean_predicted_score"],
                non_empty_bins["mean_true_label"],
                marker="o",
                label="Observed",
            )
        calibration_ax.set_title("Calibration")
        calibration_ax.set_xlabel("Mean predicted score")
        calibration_ax.set_ylabel("Mean true label")
        calibration_ax.set_xlim(0.0, 1.0)
        calibration_ax.set_ylim(0.0, 1.0)
        calibration_ax.legend()
        figures["calibration"] = calibration_fig

        distribution_fig, distribution_ax = plt.subplots()
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        distribution_ax.hist(
            self.predicted_scores,
            bins=bins,
            alpha=0.6,
            label="Predicted scores",
        )
        distribution_ax.hist(
            self.true_labels,
            bins=bins,
            alpha=0.6,
            label="True labels",
        )
        distribution_ax.set_title("Score Distribution")
        distribution_ax.set_xlabel("Score")
        distribution_ax.set_ylabel("Count")
        distribution_ax.legend()
        figures["score_distribution"] = distribution_fig

        return figures

    def to_html(self) -> str:
        """Return a single HTML document containing all report sections."""

        frames = self.to_dataframe()
        sections = [
            "<html>",
            "<body>",
            "<h1>Alignment Report</h1>",
            f"<p>Examples: {len(self.predicted_scores)}</p>",
            f"<p>Threshold: {self.threshold:g}</p>",
        ]

        section_titles = [
            ("summary", "Summary"),
            ("confusion_matrix", "Confusion Matrix"),
            ("calibration", "Calibration"),
            ("score_distribution", "Score Distribution"),
            ("worst_misses", "Worst Misses"),
            ("difficulty_breakdown", "Difficulty Breakdown"),
        ]
        for key, title in section_titles:
            sections.append(f"<h2>{html.escape(title)}</h2>")
            sections.append(frames[key].to_html(index=False))

        sections.extend(["</body>", "</html>"])
        return "\n".join(sections)
