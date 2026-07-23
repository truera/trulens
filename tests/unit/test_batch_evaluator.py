"""Tests for batch / offline evaluation (`trulens.core.batch.BatchEvaluator`)."""

import pandas as pd
import pytest
from trulens.core import BatchEvaluator
from trulens.core import Metric
from trulens.core import Selector

# --- Metric implementations used across tests (module-level so they serialize) ---


def word_overlap(query: str, answer: str):
    """Return overlap fraction plus metadata (score, meta) tuple."""
    q = set(query.lower().split())
    a = set(answer.lower().split())
    score = len(q & a) / max(len(q), 1)
    return score, {"overlap": sorted(q & a)}


def length_ratio(response: str) -> float:
    """Return a plain float score (no metadata)."""
    return min(len(response) / 100.0, 1.0)


def contains_answer(context: str, answer: str) -> float:
    """Score a single context string against the answer."""
    return 1.0 if answer.lower() in context.lower() else 0.0


def num_contexts(context: list) -> float:
    """Score using the whole list of contexts at once."""
    return float(len(context))


def custom_quality(doc: str, summary: str) -> float:
    """Custom metric with arbitrary argument names."""
    return 1.0 if summary.lower() in doc.lower() else 0.0


def _overlap_metric() -> Metric:
    return Metric(
        name="overlap",
        implementation=word_overlap,
        selectors={
            "query": Selector.from_column("q"),
            "answer": Selector.from_column("a"),
        },
    )


# --- Selector.from_column -----------------------------------------------------


def test_from_column_sets_dataset_column():
    sel = Selector.from_column("query")
    assert sel.dataset_column == "query"
    assert sel.is_dataset_selector is True
    # A span selector is not a dataset selector.
    span_sel = Selector.select_record_input()
    assert span_sel.is_dataset_selector is False


def test_from_column_collect_list_flag():
    sel = Selector.from_column("contexts", collect_list=False)
    assert sel.collect_list is False
    assert sel.dataset_column == "contexts"


def test_dataset_column_rejects_span_fields():
    with pytest.raises(ValueError):
        Selector(dataset_column="q", span_attribute="something")


# --- BatchEvaluator construction / validation ---------------------------------


def test_requires_at_least_one_metric():
    with pytest.raises(ValueError):
        BatchEvaluator(metrics=[])


def test_rejects_non_dataset_selectors():
    metric = Metric(
        name="bad",
        implementation=word_overlap,
        selectors={
            "query": Selector.select_record_input(),  # span selector
            "answer": Selector.from_column("a"),
        },
    )
    with pytest.raises(ValueError, match="not a dataset selector"):
        BatchEvaluator(metrics=[metric])


# --- evaluate() over different input types ------------------------------------


def test_evaluate_dataframe():
    ev = BatchEvaluator(metrics=[_overlap_metric()])
    df = pd.DataFrame({
        "q": ["the fox runs", "hello world"],
        "a": ["fox runs fast", "goodbye"],
    })
    res = ev.evaluate(df)

    # Result columns present, original columns preserved.
    for col in ["q", "a", "overlap", "overlap_explanation", "overlap_latency"]:
        assert col in res.columns
    assert len(res) == 2
    assert res["overlap"].iloc[0] == pytest.approx(2 / 3)
    assert res["overlap"].iloc[1] == pytest.approx(0.0)
    assert res["overlap_explanation"].iloc[0] == {"overlap": ["fox", "runs"]}
    assert (res["overlap_latency"] >= 0).all()


def test_evaluate_list_of_dicts():
    ev = BatchEvaluator(metrics=[_overlap_metric()])
    data = [
        {"q": "the fox", "a": "the fox"},
        {"q": "abc", "a": "xyz"},
    ]
    res = ev.evaluate(data)
    assert res["overlap"].iloc[0] == pytest.approx(1.0)
    assert res["overlap"].iloc[1] == pytest.approx(0.0)


def test_single_mapping_is_rejected():
    ev = BatchEvaluator(metrics=[_overlap_metric()])
    with pytest.raises(TypeError):
        ev.evaluate({"q": "x", "a": "x"})  # a single row, not a list


# --- column_map ---------------------------------------------------------------


def test_column_map_renames_columns():
    ev = BatchEvaluator(metrics=[_overlap_metric()])
    data = [{"user_question": "the fox", "a": "the fox"}]
    res = ev.evaluate(data, column_map={"user_question": "q"})
    assert res["overlap"].iloc[0] == pytest.approx(1.0)


# --- list-valued columns / collect_list semantics -----------------------------


def test_collect_list_false_aggregates_per_item():
    metric = Metric(
        name="ctx",
        implementation=contains_answer,
        selectors={
            "context": Selector.from_column("contexts", collect_list=False),
            "answer": Selector.from_column("answer"),
        },
    )
    ev = BatchEvaluator(metrics=[metric])
    data = [
        {"contexts": ["paris is nice", "berlin", "madrid"], "answer": "paris"}
    ]
    res = ev.evaluate(data)
    # 1 of 3 contexts contains "paris" -> mean = 1/3
    assert res["ctx"].iloc[0] == pytest.approx(1 / 3)


def test_collect_list_true_passes_whole_list():
    metric = Metric(
        name="nctx",
        implementation=num_contexts,
        selectors={
            "context": Selector.from_column("contexts", collect_list=True),
        },
    )
    ev = BatchEvaluator(metrics=[metric])
    data = [{"contexts": ["a", "b", "c"]}]
    res = ev.evaluate(data)
    assert res["nctx"].iloc[0] == pytest.approx(3.0)


# --- metric returning a plain float, multiple metrics -------------------------


def test_plain_float_and_multiple_metrics():
    m1 = _overlap_metric()
    m2 = Metric(
        name="length",
        implementation=length_ratio,
        selectors={"response": Selector.from_column("a")},
    )
    ev = BatchEvaluator(metrics=[m1, m2])
    df = pd.DataFrame({"q": ["hi there"], "a": ["hi there friend"]})
    res = ev.evaluate(df)
    assert "overlap" in res.columns
    assert "length" in res.columns
    assert res["length"].iloc[0] == pytest.approx(
        len("hi there friend") / 100.0
    )
    # No metadata -> explanation is an empty dict.
    assert res["length_explanation"].iloc[0] == {}


def test_custom_metric_arbitrary_arg_names():
    metric = Metric(
        name="custom_quality",
        implementation=custom_quality,
        selectors={
            "doc": Selector.from_column("original_document"),
            "summary": Selector.from_column("generated_summary"),
        },
    )
    ev = BatchEvaluator(metrics=[metric])
    data = [
        {
            "original_document": "The quick brown fox",
            "generated_summary": "brown fox",
        }
    ]
    res = ev.evaluate(data)
    assert res["custom_quality"].iloc[0] == pytest.approx(1.0)


def test_duplicate_metric_names_are_disambiguated():
    ev = BatchEvaluator(metrics=[_overlap_metric(), _overlap_metric()])
    df = pd.DataFrame({"q": ["a b"], "a": ["a b"]})
    res = ev.evaluate(df)
    assert "overlap" in res.columns
    assert "overlap_1" in res.columns
    assert res["overlap"].iloc[0] == pytest.approx(1.0)
    assert res["overlap_1"].iloc[0] == pytest.approx(1.0)


# --- parallel vs serial equivalence -------------------------------------------


def test_parallel_matches_serial():
    df = pd.DataFrame({
        "q": [f"word{i} common" for i in range(20)],
        "a": [f"word{i} common extra" for i in range(20)],
    })
    serial = BatchEvaluator(
        metrics=[_overlap_metric()], max_workers=1
    ).evaluate(df)
    parallel = BatchEvaluator(
        metrics=[_overlap_metric()], max_workers=8
    ).evaluate(df)
    pd.testing.assert_series_equal(
        serial["overlap"], parallel["overlap"], check_names=False
    )


# --- missing / None handling --------------------------------------------------


def test_missing_column_raises():
    ev = BatchEvaluator(metrics=[_overlap_metric()])
    with pytest.raises(KeyError):
        ev.evaluate([{"q": "only q, no a"}])


def test_ignore_none_values_skips_row():
    metric = Metric(
        name="overlap",
        implementation=word_overlap,
        selectors={
            "query": Selector.from_column("q", ignore_none_values=True),
            "answer": Selector.from_column("a", ignore_none_values=True),
        },
    )
    ev = BatchEvaluator(metrics=[metric])
    data = [
        {"q": "the fox", "a": "the fox"},
        {"q": "the fox", "a": None},
    ]
    res = ev.evaluate(data)
    assert res["overlap"].iloc[0] == pytest.approx(1.0)
    # Second row skipped -> no score (None/NaN).
    assert pd.isna(res["overlap"].iloc[1])


def test_empty_dataset_returns_empty_frame():
    ev = BatchEvaluator(metrics=[_overlap_metric()])
    res = ev.evaluate([])
    assert len(res) == 0
    assert "overlap" in res.columns
