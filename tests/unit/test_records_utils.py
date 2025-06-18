from datetime import datetime

import pandas as pd
import pytest
from trulens.dashboard.utils.records_utils import _filter_duplicate_span_calls
from trulens.dashboard.utils.records_utils import _filter_eval_calls_by_root
from trulens.dashboard.utils.records_utils import _identify_span_types

# Test data for OTel spans
OTEL_EVAL_ROOT = {
    "span_type": "EVAL_ROOT",
    "eval_root_id": "root1",
    "timestamp": datetime.now().isoformat(),
    "args_span_id": "span1",
    "args_span_attribute": "attr1",
}

OTEL_EVAL = {
    "span_type": "EVAL",
    "eval_root_id": "root1",
    "timestamp": datetime.now().isoformat(),
    "args": {"input": "test"},
    "ret": 0.8,
    "meta": {"feedback": "test"},
}

# Test data for legacy spans
LEGACY_EVAL = {
    "args": {"input": "test"},
    "ret": 0.8,
    "meta": {"feedback": "test"},
}


def test_identify_span_types_otel():
    """Test identifying span types with OTel tracing enabled."""
    calls = [OTEL_EVAL_ROOT, OTEL_EVAL]
    eval_root_calls, eval_calls = _identify_span_types(calls)

    assert len(eval_root_calls) == 1
    assert len(eval_calls) == 1
    assert eval_root_calls[0]["span_type"] == "EVAL_ROOT"
    assert eval_calls[0]["span_type"] == "EVAL"


def test_identify_span_types_legacy():
    """Test identifying span types with legacy spans."""
    calls = [LEGACY_EVAL]
    eval_root_calls, eval_calls = _identify_span_types(calls)

    assert len(eval_root_calls) == 0
    assert len(eval_calls) == 1
    assert "args" in eval_calls[0]
    assert "ret" in eval_calls[0]
    assert "meta" in eval_calls[0]


def test_filter_eval_calls_by_root():
    """Test filtering eval calls by root spans."""
    eval_root_calls = [OTEL_EVAL_ROOT]
    eval_calls = [OTEL_EVAL]

    filtered_calls = _filter_eval_calls_by_root(eval_root_calls, eval_calls)
    assert len(filtered_calls) == 1
    assert filtered_calls[0]["eval_root_id"] == "root1"


def test_filter_eval_calls_by_root_no_roots():
    """Test filtering eval calls when no root spans exist."""
    eval_root_calls = []
    eval_calls = [OTEL_EVAL]

    filtered_calls = _filter_eval_calls_by_root(eval_root_calls, eval_calls)
    assert len(filtered_calls) == 1
    assert filtered_calls == eval_calls


def test_filter_duplicate_span_calls():
    """Test filtering duplicate span calls."""
    # Create test data with duplicates
    data = {
        "eval_root_id": ["root1", "root2", "root1"],
        "timestamp": [
            "2023-01-01T01:00:00",
            "2023-02-01T02:00:00",
            "2023-03-01T03:00:00",
        ],
        "args_span_id": ["span1", "span1", "span1"],
        "args_span_attribute": ["attr1", "attr1", "attr1"],
    }
    df = pd.DataFrame(data)

    filtered_df = _filter_duplicate_span_calls(df)
    assert len(filtered_df) == 1
    assert "eval_root_id" in filtered_df.columns
    assert "args_span_id" not in filtered_df.columns
    assert "args_span_attribute" not in filtered_df.columns
    assert "timestamp" not in filtered_df.columns


def test_filter_duplicate_span_calls_missing_columns():
    """Test filtering duplicate span calls with missing columns."""
    # Create test data without required columns
    data = {
        "eval_root_id": ["root1", "root2"],
        "other_column": ["value1", "value2"],
    }
    df = pd.DataFrame(data)
    with pytest.raises(
        KeyError, match="Required columns missing: {'timestamp'}"
    ):
        _filter_duplicate_span_calls(df)


def test_filter_duplicate_span_calls_empty():
    """Test filtering duplicate span calls with empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(
        KeyError,
        match="Required columns missing",
    ):
        _filter_duplicate_span_calls(df)


def test_identify_span_types_malformed():
    """Test _identify_span_types with malformed input (missing fields)."""
    malformed = {"span_type": "EVAL"}  # missing eval_root_id, args, ret, meta
    eval_root_calls, eval_calls = _identify_span_types([malformed])
    # Should not raise, but may not classify as valid eval
    assert isinstance(eval_root_calls, list)
    assert isinstance(eval_calls, list)


def test_identify_span_types_multiple():
    """Test _identify_span_types with multiple EVAL_ROOT and EVAL spans."""
    calls = [
        dict(OTEL_EVAL_ROOT),
        dict(OTEL_EVAL_ROOT, eval_root_id="root2"),
        dict(OTEL_EVAL),
        dict(OTEL_EVAL, eval_root_id="root2"),
    ]
    eval_root_calls, eval_calls = _identify_span_types(calls)
    assert len(eval_root_calls) == 2
    assert len(eval_calls) == 2


def test_filter_eval_calls_by_root_missing_eval_root_id():
    """Test _filter_eval_calls_by_root with EVAL call missing eval_root_id."""
    eval_root_calls = [dict(OTEL_EVAL_ROOT)]
    eval_calls = [dict(OTEL_EVAL)]
    del eval_calls[0]["eval_root_id"]
    with pytest.raises(KeyError, match="eval_root_id"):
        _filter_eval_calls_by_root(eval_root_calls, eval_calls)


def test_filter_eval_calls_by_root_non_matching_eval_root_id():
    """Test _filter_eval_calls_by_root with EVAL call referencing non-existent root."""
    eval_root_calls = [dict(OTEL_EVAL_ROOT)]
    eval_calls = [dict(OTEL_EVAL, eval_root_id="not_a_root")]  # not in roots
    filtered = _filter_eval_calls_by_root(eval_root_calls, eval_calls)
    assert filtered == []


def test_filter_eval_calls_by_root_missing_eval_root_id_in_root():
    """Test _filter_eval_calls_by_root with EVAL_ROOT missing eval_root_id."""
    eval_root_calls = [dict(OTEL_EVAL_ROOT)]
    del eval_root_calls[0]["eval_root_id"]
    eval_calls = [dict(OTEL_EVAL)]
    with pytest.raises(KeyError, match="eval_root_id"):
        _filter_eval_calls_by_root(eval_root_calls, eval_calls)


def test_filter_duplicate_span_calls_multiple_groups():
    """Test _filter_duplicate_span_calls with multiple groups, each with duplicates."""
    data = {
        "eval_root_id": ["root1", "root2", "root3", "root4"],
        "timestamp": [
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ],
        "args_span_id": ["span1", "span1", "span2", "span2"],
        "args_span_attribute": ["attr1", "attr1", "attr2", "attr2"],
    }
    df = pd.DataFrame(data)
    filtered_df = _filter_duplicate_span_calls(df)
    # Should keep only the most recent for each group
    assert len(filtered_df) == 2


def test_filter_duplicate_span_calls_invalid_timestamps():
    """Test _filter_duplicate_span_calls with invalid timestamps."""
    data = {
        "eval_root_id": ["root1", "root2"],
        "timestamp": [None, None],
        "args_span_id": ["span1", "span1"],
        "args_span_attribute": ["attr1", "attr1"],
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="timestamp"):
        _filter_duplicate_span_calls(df)


def test_filter_duplicate_span_calls_all_columns_missing():
    """Test _filter_duplicate_span_calls with all required columns missing."""
    data = {"foo": [1, 2], "bar": [3, 4]}
    df = pd.DataFrame(data)
    with pytest.raises(KeyError, match="eval_root_id"):
        _filter_duplicate_span_calls(df)
