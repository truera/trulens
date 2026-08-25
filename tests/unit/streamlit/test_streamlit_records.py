import pandas as pd
from trulens.dashboard.tabs.Records import _conversation_metric_row
from trulens.dashboard.tabs.Records import _conversation_turn_html
from trulens.dashboard.tabs.Records import _first_token_latency_ms
from trulens.dashboard.tabs.Records import _partition_feedback_scopes
from trulens.dashboard.tabs.Records import _preprocess_df
from trulens.dashboard.tabs.Records import _split_conversation_turns
from trulens.otel.semconv.trace import SpanAttributes


def _records() -> pd.DataFrame:
    reasons = ["evaluated", "not_sampled", "throttled", None, "future_reason"]
    return pd.DataFrame({
        "app_id": ["app"] * 5,
        "app_name": ["Test App"] * 5,
        "app_version": ["v1"] * 5,
        "record_id": [f"record-{i}" for i in range(5)],
        "input": [f"input {i}" for i in range(5)],
        "output": [f"output {i}" for i in range(5)],
        "ts": pd.date_range("2026-07-01", periods=5),
        "eval_decision_reason": reasons,
        "sample_rate": [0.1, 0.1, 0.1, None, 0.1],
    })


def test_preprocess_adds_online_eval_columns():
    result = _preprocess_df(_records())

    assert result["online_eval_status"].tolist() == [
        "Unknown",
        "Not configured",
        "Skipped · throttled",
        "Skipped · sampled out",
        "Selected",
    ]
    assert result["sample_rate_display"].tolist() == [
        "10%",
        "—",
        "10%",
        "10%",
        "10%",
    ]


def test_preprocess_filters_skipped_records():
    result = _preprocess_df(_records(), online_eval_filter="Skipped")

    assert set(result["record_id"]) == {"record-1", "record-2"}


def test_conversation_bubbles_use_theme_colors():
    html = _conversation_turn_html(
        record_id="record-1",
        turn_label="Conversation turn",
        border="var(--primary-color)",
        user_text="Question",
        assistant_text="Answer",
    )

    assert "var(--text-color)" in html
    assert "var(--secondary-background-color)" in html
    assert "var(--background-color)" in html
    assert "#2a2a3d" not in html
    assert "#1a3a2a" not in html


def test_partition_feedback_scopes_from_selector_metadata():
    records = pd.DataFrame({
        "Conversation Completeness": [None, 1.0],
        "Conversation Completeness_calls": [
            None,
            [
                {
                    "span_type": "eval_root",
                    "args_span_attribute": {"records": "conversation.records"},
                }
            ],
        ],
        "Answer Relevance": [0.8, None],
        "Answer Relevance_calls": [[{"span_type": "eval"}], None],
    })

    conversation_metrics, turn_metrics = _partition_feedback_scopes(
        records,
        ["Conversation Completeness", "Answer Relevance"],
    )

    assert conversation_metrics == ["Conversation Completeness"]
    assert turn_metrics == ["Answer Relevance"]


def test_conversation_metric_row_uses_score_owner():
    records = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Conversation Completeness": [None, 1.0],
        "record_id": ["turn-1", "turn-2"],
    })

    result = _conversation_metric_row(records, "Conversation Completeness")

    assert result["record_id"] == "turn-2"
    assert result["Conversation Completeness"] == 1.0


def test_split_long_conversation_collapses_middle_turns():
    records = pd.DataFrame({"record_id": [f"turn-{i}" for i in range(12)]})

    first, middle, last = _split_conversation_turns(records)

    assert first["record_id"].tolist() == ["turn-0"]
    assert middle["record_id"].tolist() == [f"turn-{i}" for i in range(1, 11)]
    assert last["record_id"].tolist() == ["turn-11"]


def test_split_short_conversation_keeps_all_turns_visible():
    records = pd.DataFrame({"record_id": ["turn-0", "turn-1"]})

    first, middle, last = _split_conversation_turns(records)

    assert first["record_id"].tolist() == ["turn-0", "turn-1"]
    assert middle.empty
    assert last.empty


def test_first_token_latency_ms_finds_streaming_generation_span():
    spans = [
        {"record_attributes": {SpanAttributes.SPAN_TYPE: "record_root"}},
        {
            "record_attributes": {
                SpanAttributes.SPAN_TYPE: "generation",
                SpanAttributes.GENERATION.IS_STREAMING: True,
                SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS: 123.4,
            },
            "start_timestamp": 1,
        },
    ]
    assert _first_token_latency_ms(spans) == 123.4


def test_first_token_latency_ms_picks_earliest_of_several_streamed_spans():
    spans = [
        {
            "record_attributes": {
                SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS: 999.0,
            },
            "start_timestamp": 5,
        },
        {
            "record_attributes": {
                SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS: 42.0,
            },
            "start_timestamp": 1,
        },
    ]
    assert _first_token_latency_ms(spans) == 42.0


def test_first_token_latency_ms_none_when_no_spans():
    assert _first_token_latency_ms(None) is None
    assert _first_token_latency_ms([]) is None


def test_first_token_latency_ms_none_for_non_streaming_record():
    spans = [
        {
            "record_attributes": {
                SpanAttributes.SPAN_TYPE: "generation",
            },
            "start_timestamp": 1,
        }
    ]
    assert _first_token_latency_ms(spans) is None
