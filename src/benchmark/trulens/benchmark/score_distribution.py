"""Score distribution diagnostics for feedback functions.

A feedback function can post a reasonable mean error yet still be useless if it
returns nearly the same score for every input. A judge that scores everything
0.4-0.6 cannot rank examples no matter how good its average looks.

[ScoreDistributionAnalyzer][trulens.benchmark.score_distribution.ScoreDistributionAnalyzer]
runs a feedback function over a golden set and reports how well the resulting
scores *discriminate*: the score histogram, a calibration curve against the
expected scores, discrimination metrics (standard deviation, unique score count,
normalized entropy) and the predicted spread per expected-score bucket. It also
flags common judge pathologies such as poor discrimination, leniency bias and
binary (bimodal) scoring. It complements the scalar metrics in
`GroundTruthAggregator` (mae, brier, ece) by diagnosing the *shape* of a judge's
output rather than a single error number.

Example:
    ```python
    from trulens.benchmark.score_distribution import ScoreDistributionAnalyzer
    from trulens.providers.openai import OpenAI

    analyzer = ScoreDistributionAnalyzer(
        feedback_fn=OpenAI().relevance,
        golden_set=my_golden_set,
    )
    report = analyzer.run()
    report.print_summary()
    report.plot()  # matplotlib histogram + calibration curve
    ```
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_NUM_BINS = 10
# A judge whose scores vary less than this (std) barely discriminates.
LOW_STD_THRESHOLD = 0.1
# Fewer distinct scores than this (rounded to 2 dp) means coarse output.
MIN_UNIQUE_SCORES = 3
# If no score falls below this, the judge may be too lenient.
LENIENCY_FLOOR = 0.2
# Mass at each extreme bin above which scoring looks binary rather than graded.
BIMODAL_EACH_EXTREME = 0.25


def _normalized_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a histogram, normalized to ``[0, 1]``.

    Args:
        counts: Per-bin counts.

    Returns:
        ``0`` when all mass is in one bin, ``1`` when it is spread uniformly
        across the bins.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


class ScoreDistributionReport:
    """Diagnostics computed from predicted (and optional expected) scores.

    Args:
        predicted: Scores produced by the feedback function, each in
            ``[0, 1]``.
        expected: Optional ground-truth scores aligned with ``predicted``;
            entries may be ``None`` when unavailable.
        num_bins: Number of histogram/calibration bins over ``[0, 1]``.
    """

    def __init__(
        self,
        predicted: Sequence[float],
        expected: Optional[Sequence[Optional[float]]] = None,
        num_bins: int = DEFAULT_NUM_BINS,
    ):
        self.predicted: np.ndarray = np.asarray(predicted, dtype=float)
        self.num_bins: int = num_bins
        if expected is None:
            self.expected: Optional[np.ndarray] = None
        else:
            self.expected = np.asarray(
                [np.nan if e is None else e for e in expected], dtype=float
            )

        self.count: int = int(self.predicted.size)
        self.mean: float = float(np.mean(self.predicted)) if self.count else 0.0
        self.std: float = float(np.std(self.predicted)) if self.count else 0.0
        self.min: float = float(np.min(self.predicted)) if self.count else 0.0
        self.max: float = float(np.max(self.predicted)) if self.count else 0.0
        self.unique_count: int = int(
            np.unique(np.round(self.predicted, 2)).size
        )
        self.counts, self.bin_edges = np.histogram(
            self.predicted, bins=num_bins, range=(0.0, 1.0)
        )
        self.entropy: float = _normalized_entropy(self.counts)
        # bin index in [0, num_bins - 1] for each predicted score
        self._bin_index = np.clip(
            np.digitize(self.predicted, self.bin_edges) - 1, 0, num_bins - 1
        )

    def histogram(self) -> List[Tuple[float, float, int]]:
        """Return the score histogram as ``(low, high, count)`` per bin."""
        edges = self.bin_edges
        return [
            (float(edges[i]), float(edges[i + 1]), int(self.counts[i]))
            for i in range(len(self.counts))
        ]

    def calibration_curve(self) -> List[Tuple[float, float, int]]:
        """Mean expected score per predicted-score bin.

        Returns:
            ``(bin_center, mean_expected, n)`` for each non-empty bin. A
            well-calibrated judge has ``mean_expected`` close to
            ``bin_center``. Empty when no expected scores were provided.
        """
        if self.expected is None:
            return []
        curve: List[Tuple[float, float, int]] = []
        for i in range(self.num_bins):
            exp = self.expected[self._bin_index == i]
            exp = exp[~np.isnan(exp)]
            if exp.size:
                center = float((self.bin_edges[i] + self.bin_edges[i + 1]) / 2)
                curve.append((center, float(np.mean(exp)), int(exp.size)))
        return curve

    def bucket_spread(self) -> Dict[str, Dict[str, float]]:
        """Predicted-score mean/std grouped by expected-score bucket.

        Buckets are ``low`` ``[0, 1/3)``, ``medium`` ``[1/3, 2/3)`` and
        ``high`` ``[2/3, 1]``, showing whether the judge separates examples
        that humans scored low, medium and high. Empty when no expected scores.
        """
        if self.expected is None:
            return {}
        buckets = {
            "low": (0.0, 1.0 / 3),
            "medium": (1.0 / 3, 2.0 / 3),
            "high": (2.0 / 3, 1.0 + 1e-9),
        }
        out: Dict[str, Dict[str, float]] = {}
        for name, (low, high) in buckets.items():
            preds = self.predicted[
                (self.expected >= low) & (self.expected < high)
            ]
            if preds.size:
                out[name] = {
                    "n": float(preds.size),
                    "predicted_mean": float(np.mean(preds)),
                    "predicted_std": float(np.std(preds)),
                }
        return out

    def flags(self) -> List[str]:
        """Human-readable warnings about judge pathologies."""
        flags: List[str] = []
        if self.count == 0:
            return flags
        if (
            self.std < LOW_STD_THRESHOLD
            or self.unique_count < MIN_UNIQUE_SCORES
        ):
            flags.append(
                f"Low discrimination: std={self.std:.3f}, "
                f"{self.unique_count} unique score(s). The judge returns "
                f"near-constant scores and cannot rank examples."
            )
        if self.min >= LENIENCY_FLOOR:
            flags.append(
                f"Possible leniency bias: no score below "
                f"{LENIENCY_FLOOR:.2f} (min={self.min:.3f})."
            )
        if (
            self.counts[0] / self.count >= BIMODAL_EACH_EXTREME
            and self.counts[-1] / self.count >= BIMODAL_EACH_EXTREME
        ):
            flags.append(
                "Bimodal distribution: a large share of scores sit at both "
                "extremes, so the judge is effectively binary, not graded."
            )
        return flags

    def print_summary(self) -> None:
        """Print a text summary of the score distribution and any flags."""
        lines = [
            "Score Distribution Report",
            "=" * 40,
            f"n={self.count}  mean={self.mean:.3f}  std={self.std:.3f}  "
            f"min={self.min:.3f}  max={self.max:.3f}",
            f"unique(2dp)={self.unique_count}  "
            f"entropy={self.entropy:.3f} (0=constant, 1=uniform)",
            "",
            "Histogram:",
        ]
        for low, high, n in self.histogram():
            lines.append(f"  [{low:.1f}-{high:.1f}) {n:>4}  {'#' * n}")
        spread = self.bucket_spread()
        if spread:
            lines.append("")
            lines.append("Predicted score by expected bucket:")
            for name in ("low", "medium", "high"):
                if name in spread:
                    s = spread[name]
                    lines.append(
                        f"  {name:>6}: mean={s['predicted_mean']:.3f} "
                        f"std={s['predicted_std']:.3f} (n={int(s['n'])})"
                    )
        flags = self.flags()
        lines.append("")
        if flags:
            lines.append("Warnings:")
            for flag in flags:
                lines.append(f"  ! {flag}")
        else:
            lines.append("No distribution pathologies detected.")
        print("\n".join(lines))

    def plot(self):  # pragma: no cover - needs matplotlib and a display
        """Plot the score histogram and calibration curve with matplotlib.

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
        curve = self.calibration_curve()
        fig, axes = plt.subplots(1, 2 if curve else 1, figsize=(11, 4))
        ax_hist = axes[0] if curve else axes
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        ax_hist.bar(centers, self.counts, width=0.9 / self.num_bins)
        ax_hist.set_title("Predicted score histogram")
        ax_hist.set_xlabel("score")
        ax_hist.set_ylabel("count")
        if curve:
            ax_cal = axes[1]
            ax_cal.plot([0, 1], [0, 1], "--", color="gray")
            ax_cal.plot([c[0] for c in curve], [c[1] for c in curve], "o-")
            ax_cal.set_title("Calibration (predicted vs expected)")
            ax_cal.set_xlabel("predicted score (bin center)")
            ax_cal.set_ylabel("mean expected score")
        fig.tight_layout()
        return fig


class ScoreDistributionAnalyzer:
    """Run a feedback function over a golden set and analyze its scores.

    Args:
        feedback_fn: The feedback function to evaluate. Called once per golden
            example and expected to return a score in ``[0, 1]`` (a trailing
            metadata value or a ``dict`` of scores is also accepted).
        golden_set: A list of dicts, each with ``query``, ``expected_response``
            and, optionally, ``expected_score``.
        num_bins: Number of histogram/calibration bins over ``[0, 1]``.
        args_fn: Optional function mapping a golden row to the positional
            arguments passed to ``feedback_fn``. Defaults to
            ``(row["query"], row["expected_response"])``.
    """

    def __init__(
        self,
        feedback_fn: Callable[..., Any],
        golden_set: List[Dict[str, Any]],
        num_bins: int = DEFAULT_NUM_BINS,
        args_fn: Optional[Callable[[Dict[str, Any]], Tuple]] = None,
    ):
        self.feedback_fn = feedback_fn
        self.golden_set = golden_set
        self.num_bins = num_bins
        self.args_fn = args_fn or self._default_args

    @staticmethod
    def _default_args(row: Dict[str, Any]) -> Tuple:
        return (row["query"], row["expected_response"])

    @staticmethod
    def _to_score(result: Any) -> float:
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, dict):
            values = [v for v in result.values() if isinstance(v, (int, float))]
            result = float(np.mean(values)) if values else float("nan")
        return float(result)

    def run(self) -> ScoreDistributionReport:
        """Run the feedback function over the golden set.

        Returns:
            A populated
            [ScoreDistributionReport][trulens.benchmark.score_distribution.ScoreDistributionReport].

        Raises:
            ValueError: If the feedback function produced no scores.
        """
        predicted: List[float] = []
        expected: List[Optional[float]] = []
        for row in self.golden_set:
            try:
                score = self._to_score(self.feedback_fn(*self.args_fn(row)))
            except Exception:
                log.exception("feedback_fn failed on a row; skipping it")
                continue
            predicted.append(score)
            expected.append(row.get("expected_score"))
        if not predicted:
            raise ValueError(
                "No scores were produced; check feedback_fn and golden_set."
            )
        return ScoreDistributionReport(predicted, expected, self.num_bins)
