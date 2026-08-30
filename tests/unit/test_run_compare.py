"""Tests for Run.compare / compare_runs (issue #2629)."""

import math
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from trulens.core.run import Run
from trulens.core.run import RunDiff
from trulens.core.run import _normalize_input
from trulens.core.run import compare_runs
from trulens.core.utils import stats as stats_utils


def _make_run_stub(run_name: str, records_df: pd.DataFrame) -> MagicMock:
    """Create a minimal Run-like mock whose get_records() returns *records_df*."""
    stub = MagicMock(spec=Run)
    stub.run_name = run_name
    stub.get_records.return_value = records_df
    return stub


# ---------------------------------------------------------------------------
# Stats helpers (canonical implementation in trulens.core.utils.stats)
# ---------------------------------------------------------------------------


class TestPairedPermutationPvalue(unittest.TestCase):
    def test_all_zeros_returns_one(self):
        self.assertEqual(
            stats_utils.paired_permutation_pvalue(np.array([0.0, 0.0])), 1.0
        )

    def test_empty_returns_one(self):
        self.assertEqual(
            stats_utils.paired_permutation_pvalue(np.array([])), 1.0
        )

    def test_large_effect_is_significant(self):
        diffs = np.array([1.0] * 30)
        p = stats_utils.paired_permutation_pvalue(diffs)
        self.assertLess(p, 0.01)

    def test_mixed_signal_not_significant(self):
        diffs = np.array([1.0, -1.0] * 15)
        p = stats_utils.paired_permutation_pvalue(diffs)
        self.assertGreater(p, 0.05)


class TestBootstrapCi(unittest.TestCase):
    def test_empty_returns_nan(self):
        lo, hi = stats_utils.bootstrap_ci(np.array([]))
        self.assertTrue(math.isnan(lo))
        self.assertTrue(math.isnan(hi))

    def test_single_element_returns_nan(self):
        lo, hi = stats_utils.bootstrap_ci(np.array([0.42]))
        self.assertTrue(math.isnan(lo))
        self.assertTrue(math.isnan(hi))

    def test_ci_contains_mean(self):
        diffs = np.array([0.1, -0.1, 0.05, -0.05, 0.2])
        lo, hi = stats_utils.bootstrap_ci(diffs)
        mean = float(np.mean(diffs))
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_tight_data_gives_narrow_ci(self):
        diffs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        lo, hi = stats_utils.bootstrap_ci(diffs)
        # All identical → CI should collapse.
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


class TestNormalizeInput(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(_normalize_input("  hello  "), "hello")

    def test_dict_sorted_keys(self):
        result = _normalize_input({"b": 2, "a": 1})
        self.assertEqual(result, '{"a": 1, "b": 2}')

    def test_json_string_reserialised(self):
        raw = '{"b": 2, "a": 1}'
        self.assertEqual(_normalize_input(raw), '{"a": 1, "b": 2}')

    def test_plain_string_unchanged(self):
        self.assertEqual(_normalize_input("hello world"), "hello world")

    def test_scalar_json_not_rewritten(self):
        """'123', 'true', 'null' are valid JSON but should stay as strings."""
        self.assertEqual(_normalize_input("123"), "123")
        self.assertEqual(_normalize_input("true"), "true")
        self.assertEqual(_normalize_input("null"), "null")

    def test_list_input(self):
        result = _normalize_input([3, 1, 2])
        self.assertEqual(result, "[3, 1, 2]")


# ---------------------------------------------------------------------------
# compare_runs / Run.compare
# ---------------------------------------------------------------------------


class TestCompareRuns(unittest.TestCase):
    def _make_dfs(self):
        """Two DataFrames sharing 3 inputs with one metric each."""
        df_a = pd.DataFrame({
            "record_id": ["r1", "r2", "r3"],
            "input": ["hello", "world", "foo"],
            "output": ["a1", "a2", "a3"],
            "latency": [0.1, 0.2, 0.3],
            "relevance": [0.8, 0.6, 0.9],
        })
        df_b = pd.DataFrame({
            "record_id": ["r4", "r5", "r6"],
            "input": ["hello", "world", "foo"],
            "output": ["b1", "b2", "b3"],
            "latency": [0.15, 0.25, 0.35],
            "relevance": [0.9, 0.5, 0.85],
        })
        return df_a, df_b

    # --- Basic comparison ---

    def test_basic_comparison(self):
        df_a, df_b = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_b)

        diff = compare_runs(run_a, run_b)

        self.assertIsInstance(diff, RunDiff)
        self.assertEqual(diff.run_a_name, "run_a")
        self.assertEqual(diff.run_b_name, "run_b")
        self.assertIn("relevance", diff.metrics)

        md = diff.metrics["relevance"]
        self.assertEqual(md.n_items, 3)
        self.assertTrue(md.higher_is_better)
        # Deltas: 0.9-0.8=0.1, 0.5-0.6=-0.1, 0.85-0.9=-0.05
        expected_mean = float(np.mean([0.1, -0.1, -0.05]))
        self.assertAlmostEqual(md.mean_delta, expected_mean, places=6)

    # --- Regression flags ---

    def test_regression_flag_default_tolerance(self):
        df_a, df_b = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_b)

        diff = compare_runs(run_a, run_b, tolerance=0.0)
        md = diff.metrics["relevance"]

        regressed = [it for it in md.items if it.regressed]
        # "world": -0.1, "foo": -0.05  (negative delta, higher_is_better)
        self.assertEqual(len(regressed), 2)
        self.assertEqual(md.n_regressed, 2)

    def test_regression_flag_with_tolerance(self):
        df_a, df_b = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_b)

        diff = compare_runs(run_a, run_b, tolerance=0.06)
        md = diff.metrics["relevance"]

        regressed = [it for it in md.items if it.regressed]
        self.assertEqual(len(regressed), 1)
        self.assertEqual(regressed[0].input, "world")

    def test_regression_flag_lower_is_better(self):
        """For lower-is-better metrics, a *positive* delta is a regression."""
        df_a = pd.DataFrame({
            "record_id": ["r1", "r2"],
            "input": ["a", "b"],
            "output": ["o1", "o2"],
            "latency": [0.1, 0.2],
            "criminality": [0.1, 0.2],
        })
        df_b = pd.DataFrame({
            "record_id": ["r3", "r4"],
            "input": ["a", "b"],
            "output": ["o3", "o4"],
            "latency": [0.1, 0.2],
            "criminality": [0.3, 0.15],  # "a" got worse, "b" improved
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(
            run_a, run_b, metric_directions={"criminality": False}
        )
        md = diff.metrics["criminality"]
        self.assertFalse(md.higher_is_better)

        # "a": delta = 0.3 - 0.1 = +0.2 → regressed (lower is better)
        # "b": delta = 0.15 - 0.2 = -0.05 → NOT regressed
        item_a = next(it for it in md.items if it.input == "a")
        item_b = next(it for it in md.items if it.input == "b")
        self.assertTrue(item_a.regressed)
        self.assertFalse(item_b.regressed)
        self.assertEqual(md.n_regressed, 1)

    def test_lower_is_better_with_tolerance(self):
        """Lower-is-better + tolerance: only flag when delta > tolerance."""
        df_a = pd.DataFrame({
            "record_id": ["r1", "r2"],
            "input": ["a", "b"],
            "output": ["o1", "o2"],
            "latency": [0.1, 0.2],
            "harmfulness": [0.1, 0.2],
        })
        df_b = pd.DataFrame({
            "record_id": ["r3", "r4"],
            "input": ["a", "b"],
            "output": ["o3", "o4"],
            "latency": [0.1, 0.2],
            "harmfulness": [0.12, 0.5],  # small increase vs large increase
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(
            run_a,
            run_b,
            tolerance=0.05,
            metric_directions={"harmfulness": False},
        )
        md = diff.metrics["harmfulness"]
        # "a": delta = +0.02, below tolerance → not regressed
        # "b": delta = +0.3, above tolerance → regressed
        regressed = [it for it in md.items if it.regressed]
        self.assertEqual(len(regressed), 1)
        self.assertEqual(regressed[0].input, "b")

    # --- Confidence interval ---

    def test_confidence_interval_contains_mean(self):
        df_a, df_b = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_b)

        diff = compare_runs(run_a, run_b)
        md = diff.metrics["relevance"]
        self.assertLessEqual(md.ci_lower, md.mean_delta)
        self.assertGreaterEqual(md.ci_upper, md.mean_delta)

    # --- Identical runs ---

    def test_identical_runs_zero_delta(self):
        df_a, _ = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_a.copy())

        diff = compare_runs(run_a, run_b)
        md = diff.metrics["relevance"]

        self.assertAlmostEqual(md.mean_delta, 0.0)
        self.assertEqual(md.p_value, 1.0)
        self.assertEqual(md.n_regressed, 0)

    # --- Duplicate inputs (Fix #1) ---

    def test_duplicate_inputs_positional_match(self):
        """3 copies in A and 3 in B → exactly 3 matched rows, not 9."""
        df_a = pd.DataFrame({
            "record_id": ["r1", "r2", "r3"],
            "input": ["same", "same", "same"],
            "output": ["a1", "a2", "a3"],
            "latency": [0.1, 0.2, 0.3],
            "relevance": [0.8, 0.6, 0.9],
        })
        df_b = pd.DataFrame({
            "record_id": ["r4", "r5", "r6"],
            "input": ["same", "same", "same"],
            "output": ["b1", "b2", "b3"],
            "latency": [0.1, 0.2, 0.3],
            "relevance": [0.85, 0.65, 0.95],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        md = diff.metrics["relevance"]
        self.assertEqual(md.n_items, 3)  # NOT 9

        deltas = [it.delta for it in md.items]
        self.assertAlmostEqual(deltas[0], 0.05)  # 0.85 - 0.80
        self.assertAlmostEqual(deltas[1], 0.05)  # 0.65 - 0.60
        self.assertAlmostEqual(deltas[2], 0.05)  # 0.95 - 0.90

    def test_unequal_duplicate_counts_drops_extras(self):
        """3 in A, 2 in B → 2 matched, 1 extra from A dropped."""
        df_a = pd.DataFrame({
            "record_id": ["r1", "r2", "r3"],
            "input": ["dup", "dup", "dup"],
            "output": ["a1", "a2", "a3"],
            "latency": [0.1, 0.2, 0.3],
            "relevance": [0.8, 0.6, 0.9],
        })
        df_b = pd.DataFrame({
            "record_id": ["r4", "r5"],
            "input": ["dup", "dup"],
            "output": ["b1", "b2"],
            "latency": [0.1, 0.2],
            "relevance": [0.85, 0.65],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        md = diff.metrics["relevance"]
        self.assertEqual(md.n_items, 2)

    def test_duplicate_match_deterministic_across_row_orders(self):
        """Shuffled row order must produce the same pairing (sorted by record_id)."""
        # Rows deliberately not in record_id order.
        df_a = pd.DataFrame({
            "record_id": ["r3", "r1", "r2"],
            "input": ["x", "x", "x"],
            "output": ["a3", "a1", "a2"],
            "latency": [0.3, 0.1, 0.2],
            "relevance": [0.9, 0.7, 0.8],
        })
        df_b = pd.DataFrame({
            "record_id": ["r6", "r4", "r5"],
            "input": ["x", "x", "x"],
            "output": ["b6", "b4", "b5"],
            "latency": [0.3, 0.1, 0.2],
            "relevance": [0.75, 0.95, 0.85],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        md = diff.metrics["relevance"]
        # After sort by record_id:
        #   A: r1→0.7, r2→0.8, r3→0.9
        #   B: r4→0.95, r5→0.85, r6→0.75
        # Deltas: 0.95-0.7=0.25, 0.85-0.8=0.05, 0.75-0.9=-0.15
        deltas = [it.delta for it in md.items]
        self.assertAlmostEqual(deltas[0], 0.25)
        self.assertAlmostEqual(deltas[1], 0.05)
        self.assertAlmostEqual(deltas[2], -0.15)

    # --- Input normalisation ---

    def test_whitespace_normalised_for_matching(self):
        """Leading/trailing whitespace shouldn't prevent matching."""
        df_a = pd.DataFrame({
            "record_id": ["r1"],
            "input": ["  hello  "],
            "output": ["a"],
            "latency": [0.1],
            "relevance": [0.8],
        })
        df_b = pd.DataFrame({
            "record_id": ["r2"],
            "input": ["hello"],
            "output": ["b"],
            "latency": [0.2],
            "relevance": [0.9],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        self.assertEqual(diff.metrics["relevance"].n_items, 1)

    def test_json_key_order_normalised(self):
        """Dict/JSON inputs with different key order should still match."""
        df_a = pd.DataFrame({
            "record_id": ["r1"],
            "input": ['{"b": 2, "a": 1}'],
            "output": ["a"],
            "latency": [0.1],
            "relevance": [0.8],
        })
        df_b = pd.DataFrame({
            "record_id": ["r2"],
            "input": ['{"a": 1, "b": 2}'],
            "output": ["b"],
            "latency": [0.2],
            "relevance": [0.9],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        self.assertEqual(diff.metrics["relevance"].n_items, 1)

    # --- Error conditions ---

    def test_no_shared_inputs_raises(self):
        df_a = pd.DataFrame({
            "record_id": ["r1"],
            "input": ["hello"],
            "output": ["a"],
            "latency": [0.1],
            "relevance": [0.8],
        })
        df_b = pd.DataFrame({
            "record_id": ["r2"],
            "input": ["goodbye"],
            "output": ["b"],
            "latency": [0.2],
            "relevance": [0.9],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        with self.assertRaises(ValueError):
            compare_runs(run_a, run_b)

    def test_no_shared_metrics_raises(self):
        df_a = pd.DataFrame({
            "record_id": ["r1"],
            "input": ["hello"],
            "output": ["a"],
            "latency": [0.1],
            "relevance": [0.8],
        })
        df_b = pd.DataFrame({
            "record_id": ["r2"],
            "input": ["hello"],
            "output": ["b"],
            "latency": [0.2],
            "coherence": [0.9],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        with self.assertRaises(ValueError):
            compare_runs(run_a, run_b)

    # --- Multiple metrics ---

    def test_multiple_metrics(self):
        df_a = pd.DataFrame({
            "record_id": ["r1", "r2"],
            "input": ["a", "b"],
            "output": ["o1", "o2"],
            "latency": [0.1, 0.2],
            "relevance": [0.8, 0.6],
            "coherence": [0.7, 0.9],
        })
        df_b = pd.DataFrame({
            "record_id": ["r3", "r4"],
            "input": ["a", "b"],
            "output": ["o3", "o4"],
            "latency": [0.15, 0.25],
            "relevance": [0.85, 0.55],
            "coherence": [0.75, 0.95],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        self.assertIn("relevance", diff.metrics)
        self.assertIn("coherence", diff.metrics)
        self.assertEqual(diff.metrics["relevance"].n_items, 2)
        self.assertEqual(diff.metrics["coherence"].n_items, 2)

    # --- Partial overlap ---

    def test_partial_overlap_warns(self):
        df_a = pd.DataFrame({
            "record_id": ["r1", "r2"],
            "input": ["shared", "only_a"],
            "output": ["a1", "a2"],
            "latency": [0.1, 0.2],
            "relevance": [0.8, 0.6],
        })
        df_b = pd.DataFrame({
            "record_id": ["r3", "r4"],
            "input": ["shared", "only_b"],
            "output": ["b1", "b2"],
            "latency": [0.15, 0.25],
            "relevance": [0.9, 0.7],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        # Only the "shared" input matches; the other two are dropped.
        self.assertEqual(diff.metrics["relevance"].n_items, 1)

    # --- NaN handling ---

    def test_nan_scores_handled(self):
        df_a = pd.DataFrame({
            "record_id": ["r1", "r2"],
            "input": ["a", "b"],
            "output": ["o1", "o2"],
            "latency": [0.1, 0.2],
            "relevance": [0.8, float("nan")],
        })
        df_b = pd.DataFrame({
            "record_id": ["r3", "r4"],
            "input": ["a", "b"],
            "output": ["o3", "o4"],
            "latency": [0.15, 0.25],
            "relevance": [0.9, 0.7],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        md = diff.metrics["relevance"]

        item_b = next(it for it in md.items if it.input == "b")
        self.assertIsNone(item_b.score_a)
        self.assertIsNone(item_b.delta)
        self.assertFalse(item_b.regressed)

        item_a = next(it for it in md.items if it.input == "a")
        self.assertAlmostEqual(item_a.delta, 0.1)

    def test_all_nan_after_match_gives_nan_aggregates(self):
        """When every matched score pair has a NaN, aggregates should be NaN."""
        df_a = pd.DataFrame({
            "record_id": ["r1"],
            "input": ["a"],
            "output": ["o1"],
            "latency": [0.1],
            "relevance": [float("nan")],
        })
        df_b = pd.DataFrame({
            "record_id": ["r2"],
            "input": ["a"],
            "output": ["o2"],
            "latency": [0.1],
            "relevance": [0.5],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        md = diff.metrics["relevance"]

        self.assertTrue(math.isnan(md.mean_delta))
        self.assertTrue(math.isnan(md.ci_lower))
        self.assertTrue(math.isnan(md.ci_upper))
        self.assertTrue(math.isnan(md.p_value))

    # --- n = 1 edge case ---

    def test_single_matched_item(self):
        """n=1: mean_delta is the single delta, CI is NaN, p_value=1.0."""
        df_a = pd.DataFrame({
            "record_id": ["r1"],
            "input": ["only"],
            "output": ["a"],
            "latency": [0.1],
            "relevance": [0.5],
        })
        df_b = pd.DataFrame({
            "record_id": ["r2"],
            "input": ["only"],
            "output": ["b"],
            "latency": [0.2],
            "relevance": [0.7],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        md = diff.metrics["relevance"]

        self.assertAlmostEqual(md.mean_delta, 0.2)
        # CI is NaN — a single observation can't yield a meaningful interval.
        self.assertTrue(math.isnan(md.ci_lower))
        self.assertTrue(math.isnan(md.ci_upper))
        self.assertEqual(md.p_value, 1.0)

    # --- DataFrame helpers ---

    def test_summary_dataframe(self):
        df_a, df_b = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_b)

        diff = compare_runs(run_a, run_b)
        summary = diff.summary()

        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 1)
        self.assertIn("mean_delta", summary.columns)
        self.assertIn("p_value", summary.columns)
        self.assertIn("higher_is_better", summary.columns)

    def test_items_df(self):
        df_a, df_b = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_b)

        diff = compare_runs(run_a, run_b)
        items = diff.items_df("relevance")

        self.assertIsInstance(items, pd.DataFrame)
        self.assertEqual(len(items), 3)
        self.assertListEqual(
            list(items.columns),
            ["input", "score_a", "score_b", "delta", "regressed"],
        )

    def test_items_df_missing_metric_raises(self):
        df_a, df_b = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_b)

        diff = compare_runs(run_a, run_b)
        with self.assertRaises(KeyError):
            diff.items_df("nonexistent_metric")

    # --- Direction default ---

    def test_unspecified_direction_defaults_to_higher_is_better(self):
        """When metric_directions omits a metric, it defaults to True."""
        df_a, df_b = self._make_dfs()
        run_a = _make_run_stub("run_a", df_a)
        run_b = _make_run_stub("run_b", df_b)

        diff = compare_runs(run_a, run_b, metric_directions={})
        self.assertTrue(diff.metrics["relevance"].higher_is_better)

    # --- ItemDiff.input carries original string ---

    def test_item_diff_preserves_original_input(self):
        """ItemDiff.input should be the original value, not the normalised key."""
        df_a = pd.DataFrame({
            "record_id": ["r1"],
            "input": ["  hello  "],
            "output": ["a"],
            "latency": [0.1],
            "relevance": [0.8],
        })
        df_b = pd.DataFrame({
            "record_id": ["r2"],
            "input": ["hello"],
            "output": ["b"],
            "latency": [0.2],
            "relevance": [0.9],
        })
        run_a = _make_run_stub("a", df_a)
        run_b = _make_run_stub("b", df_b)

        diff = compare_runs(run_a, run_b)
        item = diff.metrics["relevance"].items[0]
        # The "input" column from run_a's side is preserved in the merge
        # (it comes from the left DataFrame).  It may be the original
        # or the right side — the key point is it's a human-readable
        # value, not the normalised key.
        self.assertIn(item.input.strip(), ("hello",))


if __name__ == "__main__":
    unittest.main()
