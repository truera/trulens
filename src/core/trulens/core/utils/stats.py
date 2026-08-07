"""Lightweight statistical helpers for TruLens.

Canonical home for paired-sample tests and confidence intervals so that
``trulens-core``, ``trulens-feedback``, and ``trulens-benchmark`` all share
a single implementation.
"""

from __future__ import annotations

import numpy as np

SIGNIFICANCE_ALPHA = 0.05


def paired_permutation_pvalue(
    diffs: list[float] | np.ndarray,
    seed: int = 0,
    permutations: int = 10000,
) -> float:
    """Two-sided paired sign-flip permutation p-value for ``mean(diffs) == 0``.

    Tests whether the mean of paired score differences differs significantly
    from zero without assuming normality, by comparing the observed absolute
    mean against the distribution obtained from random sign flips.

    Args:
        diffs: Paired differences, e.g. per-example score deltas between two
            judge configurations.
        seed: Seed for the permutation random number generator.
        permutations: Number of sign-flip permutations to sample.

    Returns:
        The two-sided p-value in ``(0, 1]``.  Returns ``1.0`` when there are
        no differences or all differences are zero.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0 or np.allclose(diffs, 0.0):
        return 1.0
    observed = abs(float(np.mean(diffs)))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(permutations, n))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(perm_means >= observed) + 1) / (permutations + 1))


def bootstrap_ci(
    diffs: np.ndarray,
    alpha: float = SIGNIFICANCE_ALPHA,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of *diffs*.

    Args:
        diffs: 1-D array of paired differences.
        alpha: Significance level (default 0.05 for a 95 % CI).
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        ``(ci_lower, ci_upper)`` bounds.  Returns ``(nan, nan)`` when
        *n* < 2 (a CI is not meaningful with fewer than two observations).
    """
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diffs[indices].mean(axis=1)
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper
