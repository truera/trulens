"""
Unit tests for SamplingConfig and SamplingController.

These tests are pure-Python (no OTEL, no database) and validate the
controller's decision logic, deterministic hashing, throttling, cost
budget, and counter bookkeeping.
"""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import contextvars
import datetime
import unittest

import pytest
from trulens.core.sampling import SamplingConfig
from trulens.core.sampling import SamplingController
from trulens.core.sampling import _hash_decision
from trulens.core.sampling import ingest_eval_active


class TestSamplingConfig(unittest.TestCase):
    """Validation and defaults of SamplingConfig."""

    def test_defaults(self):
        cfg = SamplingConfig()
        assert cfg.sample_rate == 1.0
        assert cfg.throttle is None
        assert cfg.cost_budget is None

    def test_global_rate(self):
        cfg = SamplingConfig(sample_rate=0.5)
        assert cfg.sample_rate == 0.5

    def test_per_app_rates(self):
        cfg = SamplingConfig(sample_rate={"app_a": 0.1, "app_b": 0.9})
        assert cfg.sample_rate == {"app_a": 0.1, "app_b": 0.9}

    def test_rate_below_zero_raises(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SamplingConfig(sample_rate=-0.1)

    def test_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SamplingConfig(sample_rate=1.5)

    def test_per_app_rate_invalid(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SamplingConfig(sample_rate={"bad": 2.0})

    def test_throttle_zero_raises(self):
        with pytest.raises(ValueError, match="throttle must be >= 1"):
            SamplingConfig(throttle=0)

    def test_cost_budget_negative_raises(self):
        with pytest.raises(ValueError, match="cost_budget must be > 0"):
            SamplingConfig(cost_budget=-5.0)

    def test_frozen(self):
        cfg = SamplingConfig(sample_rate=0.5)
        with pytest.raises(Exception):
            cfg.sample_rate = 0.9


class TestHashDecision(unittest.TestCase):
    """Deterministic hashing produces stable, well-distributed values."""

    def test_deterministic(self):
        """Same inputs always produce the same output."""
        a = _hash_decision("record_1", "my_app")
        b = _hash_decision("record_1", "my_app")
        assert a == b

    def test_different_records_differ(self):
        a = _hash_decision("record_1", "app")
        b = _hash_decision("record_2", "app")
        assert a != b

    def test_salt_matters(self):
        """Changing the app_name salt changes the decision value."""
        a = _hash_decision("record_1", "app_a")
        b = _hash_decision("record_1", "app_b")
        assert a != b

    def test_in_range(self):
        for i in range(100):
            v = _hash_decision(f"record_{i}")
            assert 0.0 <= v < 1.0


class TestSamplingController(unittest.TestCase):
    """Controller decision logic."""

    # -- probabilistic sampling ------------------------------------------

    def test_rate_1_always_evaluates(self):
        ctrl = SamplingController(SamplingConfig(sample_rate=1.0))
        for i in range(50):
            ok, meta = ctrl.should_evaluate(f"r_{i}", "app")
            assert ok
            assert meta["eval_decision_reason"] == "evaluated"

    def test_rate_0_never_evaluates(self):
        ctrl = SamplingController(SamplingConfig(sample_rate=0.0))
        for i in range(50):
            ok, meta = ctrl.should_evaluate(f"r_{i}", "app")
            assert not ok
            assert meta["eval_decision_reason"] == "not_sampled"

    def test_rate_05_roughly_half(self):
        ctrl = SamplingController(SamplingConfig(sample_rate=0.5))
        evaluated = sum(
            ctrl.should_evaluate(f"r_{i}", "app")[0] for i in range(1000)
        )
        # Hash-based; should be roughly 500 +/- 50.
        assert 350 < evaluated < 650, f"Got {evaluated}/1000"

    def test_deterministic_replay(self):
        """Two controllers with the same config give the same decisions."""
        cfg = SamplingConfig(sample_rate=0.3)
        c1 = SamplingController(cfg)
        c2 = SamplingController(cfg)
        for i in range(100):
            rid = f"r_{i}"
            assert (
                c1.should_evaluate(rid, "app")[0]
                == c2.should_evaluate(rid, "app")[0]
            )

    # -- per-app rates ---------------------------------------------------

    def test_per_app_rate(self):
        ctrl = SamplingController(
            SamplingConfig(sample_rate={"app_a": 1.0, "app_b": 0.0})
        )
        ok_a, _meta_a = ctrl.should_evaluate("r_1", "app_a")
        ok_b, meta_b = ctrl.should_evaluate("r_1", "app_b")
        assert ok_a
        assert not ok_b
        assert meta_b["eval_decision_reason"] == "not_sampled"

    def test_per_app_unlisted_app_evaluates_normally(self):
        """Configuring sampling on one app must NOT silently disable
        eval for every other app in the session."""
        ctrl = SamplingController(SamplingConfig(sample_rate={"app_a": 0.1}))
        ok, meta = ctrl.should_evaluate("r_1", "app_b")
        assert ok, "Out-of-scope apps must still be evaluated"
        assert meta["eval_decision_reason"] == "not_configured"
        assert meta["sample_rate"] == 1.0

    def test_per_app_none_app_name_evaluates_normally(self):
        """Records with no app name evaluate normally when per-app rates set."""
        ctrl = SamplingController(SamplingConfig(sample_rate={"app_a": 1.0}))
        ok, meta = ctrl.should_evaluate("r_1", None)
        assert ok
        assert meta["eval_decision_reason"] == "not_configured"

    # -- throttle --------------------------------------------------------

    def test_throttle_limits(self):
        ctrl = SamplingController(SamplingConfig(sample_rate=1.0, throttle=3))
        results = [ctrl.should_evaluate(f"r_{i}", "app") for i in range(5)]
        evaluated = [r for ok, r in results if ok]
        throttled = [r for ok, r in results if not ok]
        assert len(evaluated) == 3
        assert len(throttled) == 2
        assert all(r["eval_decision_reason"] == "throttled" for r in throttled)

    # -- cost budget -----------------------------------------------------

    def test_cost_budget_exhausted(self):
        ctrl = SamplingController(
            SamplingConfig(sample_rate=1.0, cost_budget=1.0)
        )
        ok1, _ = ctrl.should_evaluate("r_1", "app")
        assert ok1
        ctrl.record_cost(1.5)  # Exceed budget
        ok2, meta = ctrl.should_evaluate("r_2", "app")
        assert not ok2
        assert meta["eval_decision_reason"] == "over_budget"

    def test_cost_budget_resets_daily(self):
        ctrl = SamplingController(
            SamplingConfig(sample_rate=1.0, cost_budget=1.0)
        )
        ctrl.record_cost(2.0)

        # Simulate next day by shifting the date.
        ctrl._daily_cost_date = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(days=1)
        ).date()

        ok, meta = ctrl.should_evaluate("r_1", "app")
        assert ok
        assert meta["eval_decision_reason"] == "evaluated"

    # -- gate ordering: sample -> throttle -> budget ---------------------

    def test_gate_order_sample_before_throttle(self):
        """A record rejected by sampling should not consume a throttle slot."""
        ctrl = SamplingController(SamplingConfig(sample_rate=0.0, throttle=1))
        for i in range(5):
            ok, meta = ctrl.should_evaluate(f"r_{i}", "app")
            assert not ok
            # All should be not_sampled, none throttled, because
            # sampling is checked first.
            assert meta["eval_decision_reason"] == "not_sampled"

    # -- counters --------------------------------------------------------

    def test_counters_track_decisions(self):
        ctrl = SamplingController(SamplingConfig(sample_rate=1.0, throttle=2))
        for i in range(4):
            ctrl.should_evaluate(f"r_{i}", "app")
        counters = ctrl.counters
        assert counters["evaluated"] == 2
        assert counters["throttled"] == 2

    def test_counters_snapshot_is_copy(self):
        ctrl = SamplingController(SamplingConfig())
        c1 = ctrl.counters
        ctrl.should_evaluate("r_1", "app")
        c2 = ctrl.counters
        assert c1 != c2  # Original snapshot is unchanged


class TestSamplingControllerInjection(unittest.TestCase):
    """Controller can be constructed independently for testing."""

    def test_direct_construction(self):
        ctrl = SamplingController(SamplingConfig(sample_rate=0.0))
        ok, meta = ctrl.should_evaluate("test_record")
        assert not ok
        assert meta["sample_rate"] == 0.0

    def test_forced_evaluation(self):
        """With rate=1.0 and no throttle/budget, everything evaluates."""
        ctrl = SamplingController(SamplingConfig(sample_rate=1.0))
        ok, _ = ctrl.should_evaluate("any_record", "any_app")
        assert ok


class TestOutOfScopeAppStillEvaluates(unittest.TestCase):
    """Configuring sampling on one app must NOT disable eval for others.

    This is the critical regression test for the app_not_in_scope bug:
    ``sample_rate={"prod_rag": 0.1}`` must only sample prod_rag, while
    every other app evaluates at 100%.
    """

    def test_other_app_evaluates_at_full_rate(self):
        ctrl = SamplingController(SamplingConfig(sample_rate={"prod_rag": 0.1}))
        for i in range(20):
            ok, meta = ctrl.should_evaluate(f"r_{i}", "other_app")
            assert ok, f"Record r_{i} from other_app was wrongly skipped"
            assert meta["eval_decision_reason"] == "not_configured"

    def test_scoped_app_is_sampled(self):
        ctrl = SamplingController(SamplingConfig(sample_rate={"prod_rag": 0.0}))
        ok, meta = ctrl.should_evaluate("r_1", "prod_rag")
        assert not ok
        assert meta["eval_decision_reason"] == "not_sampled"


class TestCostBudgetTrips(unittest.TestCase):
    """Verify that cost_budget actually stops evaluation when exceeded."""

    def test_budget_trips_after_exceeding(self):
        ctrl = SamplingController(
            SamplingConfig(sample_rate=1.0, cost_budget=0.50)
        )
        # First record evaluates.
        ok1, _ = ctrl.should_evaluate("r_1", "app")
        assert ok1

        # Simulate feedback cost exceeding the budget.
        ctrl.record_cost(0.60)

        # Next record should be over_budget.
        ok2, meta = ctrl.should_evaluate("r_2", "app")
        assert not ok2
        assert meta["eval_decision_reason"] == "over_budget"

    def test_budget_not_tripped_when_under(self):
        ctrl = SamplingController(
            SamplingConfig(sample_rate=1.0, cost_budget=10.0)
        )
        ctrl.record_cost(1.0)
        ok, meta = ctrl.should_evaluate("r_1", "app")
        assert ok
        assert meta["eval_decision_reason"] == "evaluated"


class TestSameRecordIdSameDecision(unittest.TestCase):
    """Hash-based sampling gives the same answer for the same record_id."""

    def test_repeated_calls_same_result(self):
        ctrl = SamplingController(SamplingConfig(sample_rate=0.5))
        for rid in ["stable_1", "stable_2", "stable_3"]:
            first_ok, _ = ctrl.should_evaluate(rid, "app")
            # Reset throttle window so it doesn't interfere.
            ctrl2 = SamplingController(SamplingConfig(sample_rate=0.5))
            second_ok, _ = ctrl2.should_evaluate(rid, "app")
            assert first_ok == second_ok, f"Mismatch for {rid}"


class TestIngestEvalActiveContextPropagation(unittest.TestCase):
    """Verify that ingest_eval_active propagates through ThreadPoolExecutor
    when using contextvars.copy_context() — the same pattern used in
    computer.py's _run_feedback_on_inputs."""

    def test_flag_visible_in_worker_with_context_copy(self):
        """Positive test: worker threads see the flag when context is copied."""
        results = []

        def worker():
            results.append(ingest_eval_active.get(False))

        token = ingest_eval_active.set(True)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(contextvars.copy_context().run, worker)
                    for _ in range(4)
                ]
                for f in as_completed(futures):
                    f.result()
        finally:
            ingest_eval_active.reset(token)

        assert all(results), f"Expected all True, got {results}"

    def test_default_value_is_false(self):
        """The default value of ingest_eval_active is False, so code
        running outside the evaluator's ingest path won't charge costs."""
        assert ingest_eval_active.get(False) is False

    def test_record_cost_charges_when_flag_set(self):
        """record_cost + ingest_eval_active = budget actually increments."""
        ctrl = SamplingController(
            SamplingConfig(sample_rate=1.0, cost_budget=10.0)
        )
        token = ingest_eval_active.set(True)
        try:
            ctrl.record_cost(5.0)
        finally:
            ingest_eval_active.reset(token)
        assert ctrl._daily_cost == 5.0

    def test_record_cost_skipped_when_flag_not_set(self):
        """Without the flag, record_cost still works (it doesn't check
        the flag — that check is in computer.py).  This test just confirms
        the controller itself is agnostic to the flag."""
        ctrl = SamplingController(
            SamplingConfig(sample_rate=1.0, cost_budget=10.0)
        )
        ctrl.record_cost(5.0)
        assert ctrl._daily_cost == 5.0


if __name__ == "__main__":
    unittest.main()
