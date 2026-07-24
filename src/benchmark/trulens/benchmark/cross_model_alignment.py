"""Cross-model alignment for feedback functions.

When you switch the model behind a judge (``llama-3.3-70b`` to ``llama-3.1-8b``,
or one provider to another) the scores can shift in ways a single accuracy
number hides. [CrossModelAlignment][trulens.benchmark.cross_model_alignment.CrossModelAlignment]
runs the same feedback method with several judges over one golden set and reports
how much they agree: pairwise Spearman correlation, mean absolute difference and
score-shift bias between every pair, plus each judge's agreement with the ground
truth, and a plain recommendation of which judges are interchangeable and which
are outliers.

The Spearman, Kendall and MAE metrics reuse
[GroundTruthAggregator][trulens.feedback.groundtruth.GroundTruthAggregator] so
there is a single source of truth for them rather than a second implementation.

Example:
    ```python
    from trulens.benchmark.cross_model_alignment import CrossModelAlignment
    from trulens.providers.litellm import LiteLLM

    judge_a = LiteLLM(model_engine="groq/llama-3.3-70b-versatile")
    judge_b = LiteLLM(model_engine="groq/llama-3.1-8b-instant")
    alignment = CrossModelAlignment(
        judges=[
            {"provider": judge_a, "name": "70b"},
            {"provider": judge_b, "name": "8b"},
        ],
        feedback_method="relevance",
        golden_set=my_golden_set,
    )
    report = alignment.run()
    report.print_matrix()
    report.plot_heatmap()
    ```
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from trulens.feedback import groundtruth as feedback_groundtruth

log = logging.getLogger(__name__)

# Pairs whose Spearman correlation is at least this are called interchangeable.
INTERCHANGEABLE_SPEARMAN = 0.9
# A judge whose mean correlation with the others is below this is an outlier.
OUTLIER_MEAN_SPEARMAN = 0.5


def _to_score(result: Any) -> float:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        values = [v for v in result.values() if isinstance(v, (int, float))]
        result = float(np.mean(values)) if values else float("nan")
    return float(result)


def _aggregator(
    reference: np.ndarray,
) -> "feedback_groundtruth.GroundTruthAggregator":
    """A GroundTruthAggregator with ``reference`` as the labels to score against."""
    return feedback_groundtruth.GroundTruthAggregator(
        true_labels=[float(x) for x in reference]
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two sequences, via GroundTruthAggregator."""
    return float(_aggregator(b).spearman_correlation([float(x) for x in a]))


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau between two sequences, via GroundTruthAggregator."""
    return float(_aggregator(b).kendall_tau([float(x) for x in a]))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error between two sequences, via GroundTruthAggregator."""
    return float(_aggregator(b).mae([float(x) for x in a]))


class CrossModelAlignmentReport:
    """Pairwise agreement between judges plus per-judge ground-truth metrics.

    Args:
        names: Judge names, one per score sequence.
        scores: Per-judge score sequences, all aligned and equal length.
        expected: Optional ground-truth scores aligned with ``scores``.
        interchangeable_spearman: Pairs whose Spearman correlation is at least
            this are reported as interchangeable.
        outlier_mean_spearman: A judge whose mean correlation with the others is
            below this is reported as an outlier.
    """

    def __init__(
        self,
        names: List[str],
        scores: List[List[float]],
        expected: Optional[List[float]] = None,
        interchangeable_spearman: float = INTERCHANGEABLE_SPEARMAN,
        outlier_mean_spearman: float = OUTLIER_MEAN_SPEARMAN,
    ):
        self.names: List[str] = list(names)
        self.scores: List[np.ndarray] = [
            np.asarray(s, dtype=float) for s in scores
        ]
        self.n: int = len(self.names)
        self.expected: Optional[np.ndarray] = (
            None if expected is None else np.asarray(expected, dtype=float)
        )
        self.interchangeable_spearman = interchangeable_spearman
        self.outlier_mean_spearman = outlier_mean_spearman

        self.spearman = np.full((self.n, self.n), np.nan)
        self.mean_abs_diff = np.full((self.n, self.n), np.nan)
        self.bias = np.full((self.n, self.n), np.nan)
        for i in range(self.n):
            for j in range(self.n):
                self.spearman[i, j] = _spearman(self.scores[i], self.scores[j])
                self.mean_abs_diff[i, j] = _mae(self.scores[i], self.scores[j])
                self.bias[i, j] = float(
                    np.mean(self.scores[i]) - np.mean(self.scores[j])
                )

    def ground_truth_metrics(self) -> Dict[str, Dict[str, float]]:
        """Per-judge agreement with the ground truth (mae, spearman, kendall).

        Empty when no expected scores were provided.
        """
        if self.expected is None:
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for name, judge_scores in zip(self.names, self.scores):
            out[name] = {
                "mae": _mae(judge_scores, self.expected),
                "spearman": _spearman(judge_scores, self.expected),
                "kendall": _kendall_tau(judge_scores, self.expected),
            }
        return out

    def recommendations(self) -> List[str]:
        """Plain-language notes on interchangeable judges and outliers."""
        recs: List[str] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.spearman[i, j] >= self.interchangeable_spearman:
                    recs.append(
                        f"{self.names[i]} and {self.names[j]} are "
                        f"interchangeable (Spearman "
                        f"{self.spearman[i, j]:.2f})."
                    )
        for i in range(self.n):
            others = [self.spearman[i, j] for j in range(self.n) if j != i]
            others = [o for o in others if not np.isnan(o)]
            if others and np.mean(others) < self.outlier_mean_spearman:
                recs.append(
                    f"{self.names[i]} is an outlier (mean Spearman "
                    f"{np.mean(others):.2f} with the other judges)."
                )
        return recs

    def print_matrix(self) -> None:
        """Print the pairwise agreement matrices and recommendations."""
        width = max((len(n) for n in self.names), default=4)
        header = " " * (width + 2) + "".join(f"{n[:8]:>9}" for n in self.names)
        lines = ["Cross-Model Alignment", "=" * 40, "", "Spearman matrix:"]
        lines.append(header)
        for i, name in enumerate(self.names):
            row = "".join(f"{self.spearman[i, j]:>9.2f}" for j in range(self.n))
            lines.append(f"{name:>{width}}  {row}")

        notable = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                bias = self.bias[i, j]
                if abs(bias) >= 0.05:
                    higher, lower = (
                        (self.names[i], self.names[j])
                        if bias > 0
                        else (self.names[j], self.names[i])
                    )
                    notable.append(
                        f"  {higher} scores {abs(bias):.2f} higher than "
                        f"{lower} on average"
                    )
        if notable:
            lines.append("")
            lines.append("Score-shift bias:")
            lines.extend(notable)

        gt = self.ground_truth_metrics()
        if gt:
            lines.append("")
            lines.append("Agreement with ground truth:")
            lines.append(f"  {'judge':>{width}}   mae  spearman  kendall")
            for name, metric in gt.items():
                lines.append(
                    f"  {name:>{width}}  {metric['mae']:.3f}  "
                    f"{metric['spearman']:>7.2f}  {metric['kendall']:>7.2f}"
                )

        recs = self.recommendations()
        lines.append("")
        if recs:
            lines.append("Recommendations:")
            for rec in recs:
                lines.append(f"  - {rec}")
        else:
            lines.append("No interchangeable judges or outliers detected.")
        print("\n".join(lines))

    def plot_heatmap(self):  # pragma: no cover - needs matplotlib and a display
        """Plot the pairwise Spearman matrix as a heatmap.

        Returns:
            The created ``matplotlib`` figure.

        Raises:
            ImportError: If ``matplotlib`` is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. Install it with "
                "`pip install matplotlib`."
            ) from e
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(self.spearman, vmin=-1, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.n))
        ax.set_xticklabels(self.names, rotation=45, ha="right")
        ax.set_yticklabels(self.names)
        for i in range(self.n):
            for j in range(self.n):
                ax.text(
                    j,
                    i,
                    f"{self.spearman[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )
        ax.set_title("Pairwise Spearman correlation")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        return fig


class CrossModelAlignment:
    """Run one feedback method with several judges and compare their scores.

    Args:
        judges: A list of dicts, each with a ``provider`` (a feedback provider
            instance) and a ``name``.
        feedback_method: The provider method to call as the judge, e.g.
            ``"relevance"``.
        golden_set: A list of dicts, each with ``query``, ``expected_response``
            and, optionally, ``expected_score``.
        args_fn: Optional function mapping a golden row to the positional
            arguments passed to each judge. Defaults to
            ``(row["query"], row["expected_response"])``.
        interchangeable_spearman: Pairs whose Spearman correlation is at least
            this are reported as interchangeable.
        outlier_mean_spearman: A judge whose mean correlation with the others is
            below this is reported as an outlier.
        skip_warn_fraction: Warn if more than this fraction of rows are skipped
            because a judge errored on them.
    """

    def __init__(
        self,
        judges: List[Dict[str, Any]],
        feedback_method: str,
        golden_set: List[Dict[str, Any]],
        args_fn: Optional[Callable[[Dict[str, Any]], Tuple]] = None,
        interchangeable_spearman: float = INTERCHANGEABLE_SPEARMAN,
        outlier_mean_spearman: float = OUTLIER_MEAN_SPEARMAN,
        skip_warn_fraction: float = 0.2,
    ):
        if len(judges) < 2:
            raise ValueError("Provide at least two judges to compare.")
        self.judges = judges
        self.feedback_method = feedback_method
        self.golden_set = golden_set
        self.args_fn = args_fn or self._default_args
        self.interchangeable_spearman = interchangeable_spearman
        self.outlier_mean_spearman = outlier_mean_spearman
        self.skip_warn_fraction = skip_warn_fraction

    @staticmethod
    def _default_args(row: Dict[str, Any]) -> Tuple:
        return (row["query"], row["expected_response"])

    def run(self) -> CrossModelAlignmentReport:
        """Score the golden set with every judge and build the report.

        Rows on which any judge errors are skipped so all judges stay aligned.

        Returns:
            A populated
            [CrossModelAlignmentReport][trulens.benchmark.cross_model_alignment.CrossModelAlignmentReport].

        Raises:
            ValueError: If no row was scored by every judge.
        """
        names = [judge["name"] for judge in self.judges]
        per_judge: Dict[str, List[float]] = {name: [] for name in names}
        expected: List[float] = []
        skipped = 0
        for row in self.golden_set:
            args = self.args_fn(row)
            row_scores: Dict[str, float] = {}
            for judge in self.judges:
                fn = getattr(judge["provider"], self.feedback_method)
                try:
                    row_scores[judge["name"]] = _to_score(fn(*args))
                except Exception:
                    log.exception(
                        "judge %s failed on a row; skipping the row",
                        judge["name"],
                    )
                    break
            if len(row_scores) != len(self.judges):
                skipped += 1
                continue
            for name, score in row_scores.items():
                per_judge[name].append(score)
            expected.append(row.get("expected_score"))

        total = len(self.golden_set)
        if total and skipped / total > self.skip_warn_fraction:
            log.warning(
                "%d of %d rows (%.0f%%) were skipped because a judge errored; "
                "the alignment is computed on the remaining %d rows.",
                skipped,
                total,
                100 * skipped / total,
                total - skipped,
            )

        scores = [per_judge[name] for name in names]
        if not scores[0]:
            raise ValueError(
                "No row was scored by every judge; check the judges and "
                "golden_set."
            )
        has_expected = all(e is not None for e in expected)
        return CrossModelAlignmentReport(
            names,
            scores,
            expected if has_expected else None,
            interchangeable_spearman=self.interchangeable_spearman,
            outlier_mean_spearman=self.outlier_mean_spearman,
        )
