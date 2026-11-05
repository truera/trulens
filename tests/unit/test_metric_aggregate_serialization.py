"""`Metric.aggregate(func)` must keep the serialized state in sync.

`Metric.__init__` mirrors a user-provided `agg` callable into the serialized
`aggregator` field (used for json/db storage and deferred feedback mode) and
computes `feedback_definition_id` from the full dump. `Metric.aggregate(func)`
only changed the in-memory `agg`, so the dump, the stored json and the
definition id kept describing the old default aggregator (issue #3026).
"""

import json
import logging
import math

import numpy as np
from trulens.core import batch as core_batch
from trulens.core.metric import metric as core_metric
from trulens.core.schema.select import Select
from trulens.core.utils import json as json_utils


def _metric(name: str = "len", **kwargs) -> core_metric.Metric:
    return core_metric.Metric(implementation=len, name=name, **kwargs)


def _aggregator_in_stored_json(
    metric: core_metric.Metric,
):
    """The json string the ORM writes into `feedback_json` (see
    `trulens.core.database.orm`), parsed back into a dict."""

    return json.loads(json_utils.json_str_of_obj(metric))["aggregator"]


def test_aggregate_sets_serialized_aggregator():
    metric = _metric().aggregate(np.max)
    ctor_metric = _metric(agg=np.max)

    dump = metric.model_dump()
    assert dump["aggregator"] is not None
    assert dump["aggregator"]["name"] == "max"
    assert dump["aggregator"] == ctor_metric.model_dump()["aggregator"]


def test_aggregate_survives_serialization_round_trip():
    metric = _metric().aggregate(np.max)

    restored = core_metric.Metric.model_validate(metric.model_dump())

    assert restored.agg.__name__ == "max"


def test_aggregate_reaches_stored_json():
    metric = _metric().on_default().aggregate(np.max)

    stored = _aggregator_in_stored_json(metric)

    assert stored is not None
    assert stored["name"] == "max"


def test_aggregate_refreshes_feedback_definition_id():
    plain = _metric()
    aggregated_max = plain.aggregate(np.max)
    aggregated_min = aggregated_max.aggregate(np.min)
    ctor_max = _metric(agg=np.max)

    assert aggregated_max.feedback_definition_id != plain.feedback_definition_id
    assert (
        aggregated_min.feedback_definition_id
        != aggregated_max.feedback_definition_id
    )
    # Metrics that differ only by how the aggregator was supplied must agree:
    assert (
        aggregated_max.feedback_definition_id == ctor_max.feedback_definition_id
    )


def test_aggregate_unserializable_warns_like_constructor(caplog):
    def local_agg(scores):
        return float(np.max(scores))

    # Mimics a function defined in a run-as-script module, which the
    # constructor's serializer rejects as documented at `Metric.__init__`.
    local_agg.__module__ = "__main__"

    with caplog.at_level(logging.WARNING):
        metric = _metric().aggregate(local_agg)

    assert "Cannot serialize aggregator" in caplog.text
    # Same fallback as the constructor: stored without a loadable reference.
    dump = metric.model_dump()
    assert dump["aggregator"] is not None
    assert dump["aggregator"]["name"] == "local_agg"


def test_aggregate_does_not_mutate_original():
    plain = _metric()
    plain_id = plain.feedback_definition_id

    aggregated = plain.aggregate(np.max)

    assert plain.model_dump()["aggregator"] is None
    assert plain.agg.__name__ == "mean"
    assert plain.feedback_definition_id == plain_id
    assert aggregated is not plain


def test_batch_aggregate_matches_ctor_aggregator_for_raising_agg():
    def bad_agg(scores):
        raise ValueError("boom")

    ctor_metric = _metric(name="bad", agg=bad_agg)
    aggregate_metric = _metric(name="bad").aggregate(bad_agg)

    ctor_result = core_batch.BatchEvaluator._aggregate_scores(
        ctor_metric, [0.5, 0.5]
    )
    # Before the fix this raised TypeError naming numpy.mean because the
    # copy's serialized aggregator stayed None, i.e. "user not configured".
    aggregate_result = core_batch.BatchEvaluator._aggregate_scores(
        aggregate_metric, [0.5, 0.5]
    )

    # The aggregate slot holds a nan, so compare the fields, not the tuples.
    assert math.isnan(ctor_result[0])
    assert math.isnan(aggregate_result[0])
    assert aggregate_result[1] == ctor_result[1] == "ValueError: boom"


def test_aggregate_keeps_a_supplied_feedback_definition_id():
    """A caller-supplied id stays put; only a derived id is refreshed."""

    pinned = core_metric.Metric(
        implementation=len,
        name="len",
        feedback_definition_id="pinned-by-caller",
    )
    assert pinned.feedback_definition_id == "pinned-by-caller"

    aggregated = pinned.aggregate(np.max)

    assert aggregated.feedback_definition_id == "pinned-by-caller"


def test_aggregate_after_on_refreshes_feedback_definition_id(monkeypatch):
    """A builder such as `on()` must not stop `aggregate()` from refreshing.

    Regression for the review on #3030: `on()` copies the metric without
    re-deriving the id, so the stored id no longer matches the id derived from
    the current state and the refresh was skipped. Two metrics that differ
    only by aggregator must still get their own ids.
    """

    # `Metric.on()` takes a Lens path only on the non-OTEL path; the reported
    # chain uses a Lens path.
    monkeypatch.setenv("TRULENS_OTEL_TRACING", "0")
    selector = Select.RecordCalls.retrieve.rets[:]

    chained = _metric().on(selector)
    aggregated_max = chained.aggregate(np.max)
    aggregated_min = aggregated_max.aggregate(np.min)

    ctor_max = core_metric.Metric(
        implementation=len, name="len", agg=np.max, selectors={"obj": selector}
    )
    ctor_min = core_metric.Metric(
        implementation=len, name="len", agg=np.min, selectors={"obj": selector}
    )

    assert (
        aggregated_max.feedback_definition_id != chained.feedback_definition_id
    )
    assert (
        aggregated_max.feedback_definition_id == ctor_max.feedback_definition_id
    )
    assert (
        aggregated_min.feedback_definition_id == ctor_min.feedback_definition_id
    )
    assert (
        aggregated_min.feedback_definition_id
        != aggregated_max.feedback_definition_id
    )


def test_aggregate_after_on_keeps_a_supplied_feedback_definition_id(
    monkeypatch,
):
    """A builder such as `on()` must not make a supplied id get refreshed."""

    monkeypatch.setenv("TRULENS_OTEL_TRACING", "0")

    pinned = core_metric.Metric(
        implementation=len,
        name="len",
        feedback_definition_id="pinned-by-caller",
    )

    aggregated = pinned.on(Select.RecordCalls.retrieve.rets[:]).aggregate(
        np.max
    )

    assert aggregated.feedback_definition_id == "pinned-by-caller"
