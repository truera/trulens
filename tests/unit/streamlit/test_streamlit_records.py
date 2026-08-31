import pandas as pd
from trulens.dashboard.tabs.Records import _build_thread_grid_options
from trulens.dashboard.tabs.Records import _build_thread_summary
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


def _conversation_thread_records() -> pd.DataFrame:
    """A two-turn thread whose conversation-scoped Topic Adherence metric is
    recorded as 1.0 on one turn and 0.0 on the latest turn."""
    base = {
        "app_id": ["app", "app"],
        "app_name": ["Test App", "Test App"],
        "app_version": ["v1", "v1"],
        "conversation_id": ["conv-1", "conv-1"],
        "record_id": ["turn-1", "turn-2"],
        "input": ["hi", "still off topic"],
        "output": ["reply", "reply"],
        "ts": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "total_tokens": [1, 1],
        "total_cost": [0.0, 0.0],
        "cost_currency": ["USD", "USD"],
        "latency": [1.0, 1.0],
        "is_match": [True, True],
        "Topic Adherence": [1.0, 0.0],
        "Topic Adherence_calls": [
            [
                {
                    "span_type": "eval_root",
                    "args_span_attribute": {"records": "conversation.records"},
                }
            ],
            [
                {
                    "span_type": "eval_root",
                    "args_span_attribute": {"records": "conversation.records"},
                }
            ],
        ],
    }
    return pd.DataFrame(base)


def test_thread_summary_conversation_metric_matches_detail_view():
    """Regression for #2725: the thread table must not average a
    conversation-scoped metric across the thread's records. Averaging [1.0, 0.0]
    surfaced 0.5 in the table while the conversation-metrics detail view showed
    0.0 for the same thread. Both must agree on the latest recorded value."""
    df = _conversation_thread_records()

    conversation_metrics, _ = _partition_feedback_scopes(
        df, ["Topic Adherence"]
    )
    assert conversation_metrics == ["Topic Adherence"]

    detail_value = _conversation_metric_row(df, "Topic Adherence")[
        "Topic Adherence"
    ]
    assert detail_value == 0.0

    summary = _build_thread_summary(df, ["Topic Adherence"])
    assert len(summary) == 1
    table_value = summary.iloc[0]["Topic Adherence"]

    # Must be the latest recorded value (0.0), not the mean (0.5).
    assert table_value == detail_value
    assert table_value == 0.0


def test_thread_summary_turn_metric_still_averaged():
    """Turn-scoped metrics must keep their per-turn mean; only conversation
    metrics change."""
    df = _conversation_thread_records()
    df = df.drop(columns=["Topic Adherence", "Topic Adherence_calls"])
    df["Answer Relevance"] = [1.0, 0.0]
    df["Answer Relevance_calls"] = [
        [{"span_type": "eval"}],
        [{"span_type": "eval"}],
    ]

    summary = _build_thread_summary(df, ["Answer Relevance"])
    assert summary.iloc[0]["Answer Relevance"] == 0.5


def test_thread_grid_header_labels_conversation_vs_turn_metrics():
    """Regression for the #2725 follow-up: conversation-scoped metric columns
    show the latest value per thread, so their header must read "(latest)",
    while turn-scoped metrics keep "(avg)"."""
    summary = pd.DataFrame({
        "thread_key": ["t1"],
        "app_name": ["a"],
        "app_version": ["v1"],
        "app_id": ["id"],
        "num_messages": [2],
        "matched_turn_count": [1],
        "total_turn_count": [2],
        "first_input": ["q"],
        "last_output": ["o"],
        "start_ts": ["2026-01-01"],
        "ts": ["2026-01-02"],
        "total_tokens": [1],
        "total_cost": [0.0],
        "cost_currency": ["USD"],
        "latency": [1.0],
        "is_thread": [True],
        "conversation_id": ["c1"],
        "record_id": [None],
        "Topic Adherence": [0.0],
        "Answer Relevance": [0.5],
    })

    options = _build_thread_grid_options(
        df=summary,
        feedback_col_names=["Topic Adherence", "Answer Relevance"],
        feedback_directions={
            "Topic Adherence": True,
            "Answer Relevance": True,
        },
        conversation_cols=["Topic Adherence"],
    )

    headers = {
        c["field"]: c.get("headerName")
        for c in options["columnDefs"]
        if c.get("field") in ("Topic Adherence", "Answer Relevance")
    }
    assert headers["Topic Adherence"] == "Topic Adherence (latest)"
    assert headers["Answer Relevance"] == "Answer Relevance (avg)"
