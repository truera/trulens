"""Tests for selecting traces for human review.

Covers the selection half of truera/trulens#2700: predicate behaviour in both
metric directions, latency and cost presets, currency separation, missing and
NaN values, composition, severity ordering, limits, and preview parity.
"""

import contextlib
import logging
import unittest

import pandas as pd
from trulens.core import review as core_review
from trulens.core.review import ReviewTargets
from trulens.core.schema import review as review_schema

RECORDS = pd.DataFrame([
    {
        "record_id": "r1",
        "Groundedness": 0.2,
        "Groundedness direction": True,
        "latency": 2.0,
        "total_cost": 0.01,
        "cost_currency": "USD",
        "app_name": "bot",
        "app_version": "v1",
        "ts": 1.0,
    },
    {
        "record_id": "r2",
        "Groundedness": 0.9,
        "Groundedness direction": True,
        "latency": 20.0,
        "total_cost": 0.20,
        "cost_currency": "USD",
        "app_name": "bot",
        "app_version": "v1",
        "ts": 2.0,
    },
    {
        "record_id": "r3",
        "Groundedness": 0.45,
        "Groundedness direction": True,
        "latency": 1.0,
        "total_cost": 5.00,
        "cost_currency": "EUR",
        "app_name": "bot",
        "app_version": "v1",
        "ts": 3.0,
    },
])


@contextlib.contextmanager
def captured_warnings(logger_name: str):
    """Collect warnings from one logger.

    Done with an explicit handler rather than `assertLogs` so that the
    assertion does not depend on whatever global logging configuration other
    test modules have already installed.
    """

    messages = []

    class _Collector(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    handler = _Collector()
    previous_level, previous_disabled = logger.level, logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.disabled = False
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled


def ids(targets):
    return [t.target_id for t in targets]


class TestScorePredicates(unittest.TestCase):
    def test_low_score(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.low_score("Groundedness", below=0.5)
        )
        self.assertEqual(sorted(ids(targets)), ["r1", "r3"])

    def test_low_score_reason_and_snapshot(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.low_score("Groundedness", below=0.5)
        )
        worst = targets[0]
        self.assertEqual(worst.target_id, "r1")
        self.assertEqual(worst.selection.selection_reason, "Groundedness < 0.5")
        self.assertEqual(worst.selection.metric_name, "Groundedness")
        self.assertEqual(worst.selection.metric_value, 0.2)
        self.assertIs(worst.selection.metric_direction, True)
        self.assertEqual(worst.selection.app_name, "bot")
        self.assertEqual(worst.selection.app_version, "v1")

    def test_high_score(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.high_score("Groundedness", above=0.5)
        )
        self.assertEqual(ids(targets), ["r2"])

    def test_worst_score_higher_is_better(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.worst_score("Groundedness", top_n=2)
        )
        # Higher is better, so the worst are the lowest scores.
        self.assertEqual(ids(targets), ["r1", "r3"])
        self.assertIn("lowest", targets[0].selection.selection_reason)

    def test_worst_score_lower_is_better(self):
        records = RECORDS.assign(**{"Groundedness direction": False})
        targets = ReviewTargets.from_records(
            records, where=ReviewTargets.worst_score("Groundedness", top_n=2)
        )
        # Lower is better, so the worst are the highest scores.
        self.assertEqual(ids(targets), ["r2", "r3"])
        self.assertIn("highest", targets[0].selection.selection_reason)

    def test_low_score_records_the_reported_direction(self):
        records = RECORDS.assign(**{"Groundedness direction": False})
        targets = ReviewTargets.from_records(
            records, where=ReviewTargets.low_score("Groundedness", below=0.5)
        )
        self.assertIs(targets[0].selection.metric_direction, False)

    def test_direction_defaults_to_higher_is_better_when_absent(self):
        records = RECORDS.drop(columns=["Groundedness direction"])
        with captured_warnings(core_review.__name__) as warnings:
            targets = ReviewTargets.from_records(
                records,
                where=ReviewTargets.worst_score("Groundedness", top_n=1),
            )
        self.assertEqual(ids(targets), ["r1"])
        self.assertIn("higher_is_better=True", "".join(warnings))

    def test_conflicting_directions_warn_and_default(self):
        records = RECORDS.assign(**{
            "Groundedness direction": [True, False, True]
        })
        with captured_warnings(core_review.__name__) as warnings:
            targets = ReviewTargets.from_records(
                records,
                where=ReviewTargets.worst_score("Groundedness", top_n=1),
            )
        self.assertEqual(ids(targets), ["r1"])
        self.assertIn("more than one direction", "".join(warnings))

    def test_missing_metric_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            ReviewTargets.from_records(
                RECORDS, where=ReviewTargets.low_score("NotAMetric", below=0.5)
            )
        self.assertIn("NotAMetric", str(caught.exception))

    def test_nan_score_never_matches(self):
        records = RECORDS.assign(Groundedness=[float("nan"), 0.9, 0.45])
        targets = ReviewTargets.from_records(
            RECORDS.assign(Groundedness=records["Groundedness"]),
            where=ReviewTargets.low_score("Groundedness", below=0.5),
        )
        # A missing score is not evidence of a problem.
        self.assertEqual(ids(targets), ["r3"])

    def test_nan_score_excluded_from_worst(self):
        records = RECORDS.assign(Groundedness=[float("nan"), 0.9, 0.45])
        targets = ReviewTargets.from_records(
            records, where=ReviewTargets.worst_score("Groundedness", top_n=3)
        )
        self.assertEqual(sorted(ids(targets)), ["r2", "r3"])

    def test_worst_score_rejects_bad_top_n(self):
        with self.assertRaises(ValueError):
            ReviewTargets.worst_score("Groundedness", top_n=0)


class TestLatencyPredicates(unittest.TestCase):
    def test_high_latency(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.high_latency(above_seconds=8)
        )
        self.assertEqual(ids(targets), ["r2"])
        self.assertEqual(targets[0].selection.selection_reason, "latency > 8s")
        self.assertEqual(targets[0].selection.latency, 20.0)

    def test_slowest(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.slowest(top_n=2)
        )
        self.assertEqual(ids(targets), ["r2", "r1"])

    def test_missing_latency_is_never_estimated(self):
        records = RECORDS.assign(latency=[float("nan"), 20.0, 1.0])
        targets = ReviewTargets.from_records(
            records, where=ReviewTargets.high_latency(above_seconds=0.5)
        )
        self.assertEqual(sorted(ids(targets)), ["r2", "r3"])

    def test_missing_latency_column_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            ReviewTargets.from_records(
                RECORDS.drop(columns=["latency"]),
                where=ReviewTargets.high_latency(above_seconds=1),
            )
        self.assertIn("latency", str(caught.exception))


class TestCostPredicates(unittest.TestCase):
    def test_high_cost_respects_currency(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.high_cost(above=0.05, currency="USD")
        )
        # r3 costs far more but is denominated in EUR.
        self.assertEqual(ids(targets), ["r2"])
        self.assertEqual(targets[0].selection.cost_currency, "USD")

    def test_high_cost_in_the_other_currency(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.high_cost(above=0.05, currency="EUR")
        )
        self.assertEqual(ids(targets), ["r3"])

    def test_most_expensive_ranks_within_one_currency(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.most_expensive(top_n=2, currency="USD")
        )
        self.assertEqual(ids(targets), ["r2", "r1"])

    def test_missing_cost_is_never_estimated(self):
        records = RECORDS.assign(total_cost=[float("nan"), 0.2, 5.0])
        targets = ReviewTargets.from_records(
            records, where=ReviewTargets.high_cost(above=0.0, currency="USD")
        )
        self.assertEqual(ids(targets), ["r2"])

    def test_missing_currency_column_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            ReviewTargets.from_records(
                RECORDS.drop(columns=["cost_currency"]),
                where=ReviewTargets.high_cost(above=0.01, currency="USD"),
            )
        self.assertIn("cost_currency", str(caught.exception))

    def test_row_with_missing_currency_is_not_eligible(self):
        records = RECORDS.assign(cost_currency=[None, "USD", "EUR"])
        targets = ReviewTargets.from_records(
            records, where=ReviewTargets.high_cost(above=0.0, currency="USD")
        )
        self.assertEqual(ids(targets), ["r2"])


class TestErrorPredicate(unittest.TestCase):
    def test_has_error(self):
        records = RECORDS.assign(error=[None, "boom", None])
        targets = ReviewTargets.from_records(
            records, where=ReviewTargets.has_error()
        )
        self.assertEqual(ids(targets), ["r2"])
        self.assertEqual(
            targets[0].selection.selection_reason, "record has an error"
        )
        self.assertEqual(targets[0].selection.priority, 1.0)

    def test_no_error_column_matches_nothing(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.has_error()
        )
        self.assertEqual(targets, [])


class TestComposition(unittest.TestCase):
    def test_or_widens(self):
        targets = ReviewTargets.from_records(
            RECORDS,
            where=(
                ReviewTargets.low_score("Groundedness", below=0.5)
                | ReviewTargets.high_latency(above_seconds=8)
            ),
        )
        self.assertEqual(sorted(ids(targets)), ["r1", "r2", "r3"])

    def test_and_narrows(self):
        targets = ReviewTargets.from_records(
            RECORDS,
            where=(
                ReviewTargets.low_score("Groundedness", below=0.5)
                & ReviewTargets.high_latency(above_seconds=0.5)
            ),
        )
        self.assertEqual(sorted(ids(targets)), ["r1", "r3"])

    def test_or_joins_reasons_for_a_row_matching_both(self):
        records = RECORDS.assign(latency=[20.0, 20.0, 1.0])
        targets = ReviewTargets.from_records(
            records,
            where=(
                ReviewTargets.low_score("Groundedness", below=0.5)
                | ReviewTargets.high_latency(above_seconds=8)
            ),
        )
        first = next(t for t in targets if t.target_id == "r1")
        self.assertIn("Groundedness < 0.5", first.selection.selection_reason)
        self.assertIn("latency > 8s", first.selection.selection_reason)
        self.assertIn(" or ", first.selection.selection_reason)

    def test_and_joins_reasons(self):
        targets = ReviewTargets.from_records(
            RECORDS,
            where=(
                ReviewTargets.low_score("Groundedness", below=0.5)
                & ReviewTargets.high_latency(above_seconds=0.5)
            ),
        )
        self.assertIn(" and ", targets[0].selection.selection_reason)

    def test_composed_priority_is_the_worst_contributor(self):
        records = RECORDS.assign(latency=[20.0, 20.0, 1.0])
        targets = ReviewTargets.from_records(
            records,
            where=(
                ReviewTargets.low_score("Groundedness", below=0.5)
                | ReviewTargets.high_latency(above_seconds=8)
            ),
        )
        first = next(t for t in targets if t.target_id == "r1")
        low_score_priority = (0.5 - 0.2) / 0.5
        self.assertGreaterEqual(first.selection.priority, low_score_priority)

    def test_composition_keeps_metric_metadata(self):
        targets = ReviewTargets.from_records(
            RECORDS,
            where=(
                ReviewTargets.low_score("Groundedness", below=0.5)
                | ReviewTargets.high_latency(above_seconds=8)
            ),
        )
        scored = next(t for t in targets if t.target_id == "r1")
        self.assertEqual(scored.selection.metric_name, "Groundedness")


class TestOrderingAndLimits(unittest.TestCase):
    def test_severity_orders_worst_first(self):
        targets = ReviewTargets.from_records(
            RECORDS,
            where=ReviewTargets.low_score("Groundedness", below=0.5),
            order_by="severity",
        )
        priorities = [t.selection.priority for t in targets]
        self.assertEqual(priorities, sorted(priorities, reverse=True))
        self.assertEqual(ids(targets), ["r1", "r3"])

    def test_created_keeps_dataframe_order(self):
        targets = ReviewTargets.from_records(
            RECORDS,
            where=ReviewTargets.low_score("Groundedness", below=0.5),
            order_by="created",
        )
        self.assertEqual(ids(targets), ["r1", "r3"])

    def test_severity_ordering_is_stable_across_row_orders(self):
        shuffled = RECORDS.iloc[::-1].reset_index(drop=True)
        first = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.low_score("Groundedness", below=0.5)
        )
        second = ReviewTargets.from_records(
            shuffled, where=ReviewTargets.low_score("Groundedness", below=0.5)
        )
        self.assertEqual(ids(first), ids(second))

    def test_limit_applies_after_ordering(self):
        targets = ReviewTargets.from_records(
            RECORDS,
            where=ReviewTargets.low_score("Groundedness", below=0.5),
            limit=1,
        )
        self.assertEqual(ids(targets), ["r1"])

    def test_unknown_ordering_is_rejected(self):
        with self.assertRaises(ValueError):
            ReviewTargets.from_records(RECORDS, order_by="whatever")

    def test_bad_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            ReviewTargets.from_records(RECORDS, limit=0)

    def test_no_predicate_selects_everything(self):
        targets = ReviewTargets.from_records(RECORDS)
        self.assertEqual(sorted(ids(targets)), ["r1", "r2", "r3"])

    def test_empty_dataframe(self):
        self.assertEqual(
            ReviewTargets.from_records(RECORDS.iloc[0:0]),
            [],
        )

    def test_non_dataframe_is_rejected(self):
        with self.assertRaises(ValueError):
            ReviewTargets.from_records([{"record_id": "r1"}])

    def test_missing_id_column_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            ReviewTargets.from_records(RECORDS.drop(columns=["record_id"]))
        self.assertIn("record_id", str(caught.exception))


class TestPreviewParity(unittest.TestCase):
    def test_preview_ids_match_materialized_ids(self):
        where = ReviewTargets.low_score(
            "Groundedness", below=0.5
        ) | ReviewTargets.high_latency(above_seconds=8)
        preview = ReviewTargets.preview(
            RECORDS, where=where, order_by="severity", limit=2
        )
        targets = ReviewTargets.from_records(
            RECORDS, where=where, order_by="severity", limit=2
        )
        # What the dashboard shows is exactly what gets queued.
        self.assertEqual(list(preview["target_id"]), ids(targets))

    def test_preview_columns(self):
        preview = ReviewTargets.preview(
            RECORDS, where=ReviewTargets.low_score("Groundedness", below=0.5)
        )
        self.assertEqual(
            list(preview.columns),
            [
                "target_id",
                "target_type",
                "selection_reason",
                "priority",
                "metric_name",
                "metric_value",
                "latency",
                "cost",
                "cost_currency",
            ],
        )


class TestSnapshotStability(unittest.TestCase):
    def test_snapshot_survives_recomputed_metrics(self):
        targets = ReviewTargets.from_records(
            RECORDS, where=ReviewTargets.low_score("Groundedness", below=0.5)
        )
        frozen = targets[0].selection.model_copy(deep=True)

        # Recompute the source metric to something else entirely.
        RECORDS.assign(Groundedness=[0.99, 0.99, 0.99])

        self.assertEqual(targets[0].selection.model_dump(), frozen.model_dump())

    def test_targets_carry_the_requested_type(self):
        targets = ReviewTargets.from_records(
            RECORDS,
            target_type=review_schema.ReviewTargetType.CONVERSATION,
            id_column="record_id",
        )
        self.assertTrue(
            all(
                t.target_type is review_schema.ReviewTargetType.CONVERSATION
                for t in targets
            )
        )

    def test_dedupe_keeps_first_occurrence(self):
        a = review_schema.ReviewTarget(target_id="r1")
        b = review_schema.ReviewTarget(target_id="r1")
        c = review_schema.ReviewTarget(target_id="r2")
        self.assertEqual(
            [t.target_id for t in core_review.dedupe_targets([a, b, c])],
            ["r1", "r2"],
        )


if __name__ == "__main__":
    unittest.main()
