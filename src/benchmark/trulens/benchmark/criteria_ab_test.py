"""A/B test two judge configurations against a golden set.

TruLens feedback functions accept `criteria` and `additional_instructions` to
tune judge behaviour, but there is no easy way to tell whether a new criteria
string actually agrees with human judgement better than the old one.
[CriteriaABTest][trulens.benchmark.criteria_ab_test.CriteriaABTest] runs two
configurations of a feedback function over the same golden set and reports which
one aligns better: side-by-side MAE / Spearman / Kendall / Brier against the
ground truth, the examples where the two disagree most, a paired significance
test on the score differences, and a winner.

Metrics are computed with numpy (no scipy dependency); significance uses a
sign-flip permutation test so the utility stays dependency-light.

Example:
    ```python
    from trulens.benchmark.criteria_ab_test import CriteriaABTest
    from trulens.providers.openai import OpenAI

    provider = OpenAI()
    test = CriteriaABTest(
        golden_set=my_golden_set,
        variant_a={"fn": provider.relevance, "name": "default"},
        variant_b={
            "fn": provider.relevance,
            "kwargs": {"criteria": "Score strictly."},
            "name": "strict",
        },
    )
    report = test.run()
    report.print_comparison()
    ```
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Number of sign-flip permutations for the significance test.
PERMUTATIONS = 10000
# Significance level below which the two variants are called different.
SIGNIFICANCE_ALPHA = 0.05


def _to_score(result: Any) -> float:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        values = [v for v in result.values() if isinstance(v, (int, float))]
        result = float(np.mean(values)) if values else float("nan")
    return float(result)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    rank_sums = np.zeros(len(counts))
    np.add.at(rank_sums, inverse, ranks)
    return (rank_sums / counts)[inverse]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a, ranks_b = _rankdata(a), _rankdata(b)
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else float("nan")


def _permutation_pvalue(diffs: np.ndarray, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for ``mean(diffs)==0``."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0 or np.allclose(diffs, 0.0):
        return 1.0
    observed = abs(float(np.mean(diffs)))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(PERMUTATIONS, n))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(perm_means >= observed) + 1) / (PERMUTATIONS + 1))


def _metrics(scores: np.ndarray, expected: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(scores - expected))),
        "spearman": _spearman(scores, expected),
        "kendall": _kendall_tau(scores, expected),
        "brier": float(np.mean((scores - expected) ** 2)),
    }


class CriteriaABTestReport:
    """Comparison of two judge configurations over the same golden set.

    Args:
        name_a: Name of variant A.
        scores_a: Variant A scores, aligned with ``scores_b``.
        name_b: Name of variant B.
        scores_b: Variant B scores.
        expected: Optional ground-truth scores aligned with the variants.
        queries: Optional per-example labels (e.g. query strings) used when
            listing the largest disagreements.
    """

    def __init__(
        self,
        name_a: str,
        scores_a: List[float],
        name_b: str,
        scores_b: List[float],
        expected: Optional[List[float]] = None,
        queries: Optional[List[str]] = None,
    ):
        self.name_a = name_a
        self.name_b = name_b
        self.scores_a = np.asarray(scores_a, dtype=float)
        self.scores_b = np.asarray(scores_b, dtype=float)
        self.expected = (
            None if expected is None else np.asarray(expected, dtype=float)
        )
        self.queries = queries
        self.diffs = self.scores_a - self.scores_b

    def metrics(self) -> Dict[str, Dict[str, float]]:
        """Per-variant agreement with ground truth. Empty if no expected."""
        if self.expected is None:
            return {}
        return {
            self.name_a: _metrics(self.scores_a, self.expected),
            self.name_b: _metrics(self.scores_b, self.expected),
        }

    def significance(self) -> Dict[str, float]:
        """Mean score difference (A - B) and its permutation p-value."""
        return {
            "mean_difference": float(np.mean(self.diffs)),
            "p_value": _permutation_pvalue(self.diffs),
        }

    def top_disagreements(self, k: int = 5) -> List[Dict[str, Any]]:
        """The ``k`` examples where the two variants differ most."""
        order = np.argsort(-np.abs(self.diffs))[:k]
        out: List[Dict[str, Any]] = []
        for i in order:
            item: Dict[str, Any] = {
                "index": int(i),
                self.name_a: float(self.scores_a[i]),
                self.name_b: float(self.scores_b[i]),
                "difference": float(self.diffs[i]),
            }
            if self.queries is not None and i < len(self.queries):
                item["query"] = self.queries[i]
            out.append(item)
        return out

    def winner(self) -> Optional[str]:
        """The variant with lower MAE vs ground truth, or None."""
        metrics = self.metrics()
        if not metrics:
            return None
        mae_a = metrics[self.name_a]["mae"]
        mae_b = metrics[self.name_b]["mae"]
        if mae_a < mae_b:
            return self.name_a
        if mae_b < mae_a:
            return self.name_b
        return None

    def print_comparison(self) -> None:
        """Print the side-by-side metrics, disagreements and the winner."""
        lines = [
            "Criteria A/B Test",
            "=" * 40,
            f"A = {self.name_a}    B = {self.name_b}    n = {len(self.diffs)}",
        ]
        metrics = self.metrics()
        if metrics:
            lines.append("")
            lines.append(
                f"  {'metric':<10}{self.name_a[:12]:>14}{self.name_b[:12]:>14}"
            )
            for key in ("mae", "spearman", "kendall", "brier"):
                a = metrics[self.name_a][key]
                b = metrics[self.name_b][key]
                lines.append(f"  {key:<10}{a:>14.3f}{b:>14.3f}")
        sig = self.significance()
        lines.append("")
        lines.append(
            f"Mean difference (A - B): {sig['mean_difference']:+.3f}  "
            f"(permutation p = {sig['p_value']:.3f})"
        )
        disagreements = self.top_disagreements()
        if disagreements:
            lines.append("")
            lines.append("Largest disagreements:")
            for d in disagreements:
                label = str(d.get("query") or f"example {d['index']}")
                label = (label[:48] + "...") if len(label) > 48 else label
                lines.append(
                    f"  {d['difference']:+.2f}  A={d[self.name_a]:.2f} "
                    f"B={d[self.name_b]:.2f}  {label}"
                )
        lines.append("")
        winner = self.winner()
        if not metrics:
            lines.append(
                "No ground truth: cannot pick a winner (showing drift only)."
            )
        elif winner is None:
            lines.append(
                "Tie: both variants have the same MAE vs ground truth."
            )
        else:
            note = (
                "significant"
                if sig["p_value"] < SIGNIFICANCE_ALPHA
                else "not significant"
            )
            lines.append(
                f"Winner: {winner} (lower MAE vs ground truth; difference "
                f"{note} at alpha={SIGNIFICANCE_ALPHA})."
            )
        print("\n".join(lines))


class CriteriaABTest:
    """Run two configurations of a feedback function and compare them.

    Args:
        golden_set: A list of dicts, each with ``query``, ``expected_response``
            and, optionally, ``expected_score``.
        variant_a: A dict with a ``fn`` (the feedback function), a ``name`` and
            optional ``kwargs`` passed to ``fn`` on every call.
        variant_b: A second variant in the same shape as ``variant_a``.
        args_fn: Optional function mapping a golden row to the positional
            arguments passed to each ``fn``. Defaults to
            ``(row["query"], row["expected_response"])``.
    """

    def __init__(
        self,
        golden_set: List[Dict[str, Any]],
        variant_a: Dict[str, Any],
        variant_b: Dict[str, Any],
        args_fn: Optional[Callable[[Dict[str, Any]], Tuple]] = None,
    ):
        for variant in (variant_a, variant_b):
            if "fn" not in variant or "name" not in variant:
                raise ValueError("each variant needs a 'fn' and a 'name'.")
        self.golden_set = golden_set
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.args_fn = args_fn or self._default_args

    @staticmethod
    def _default_args(row: Dict[str, Any]) -> Tuple:
        return (row["query"], row["expected_response"])

    def run(self) -> CriteriaABTestReport:
        """Score the golden set with both variants and build the report.

        Rows on which either variant errors are skipped so both stay aligned.

        Returns:
            A populated
            [CriteriaABTestReport][trulens.benchmark.criteria_ab_test.CriteriaABTestReport].

        Raises:
            ValueError: If no row was scored by both variants.
        """
        scores_a: List[float] = []
        scores_b: List[float] = []
        expected: List[Optional[float]] = []
        queries: List[str] = []
        for row in self.golden_set:
            args = self.args_fn(row)
            try:
                a = _to_score(
                    self.variant_a["fn"](
                        *args, **self.variant_a.get("kwargs", {})
                    )
                )
                b = _to_score(
                    self.variant_b["fn"](
                        *args, **self.variant_b.get("kwargs", {})
                    )
                )
            except Exception:
                log.exception("a variant failed on a row; skipping the row")
                continue
            scores_a.append(a)
            scores_b.append(b)
            expected.append(row.get("expected_score"))
            queries.append(str(row.get("query", "")))
        if not scores_a:
            raise ValueError(
                "No row was scored by both variants; check the variants and "
                "golden_set."
            )
        has_expected = all(e is not None for e in expected)
        return CriteriaABTestReport(
            self.variant_a["name"],
            scores_a,
            self.variant_b["name"],
            scores_b,
            expected if has_expected else None,
            queries,
        )
