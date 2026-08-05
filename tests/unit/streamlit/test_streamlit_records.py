import pandas as pd
from trulens.dashboard.tabs.Records import _conversation_metric_row
from trulens.dashboard.tabs.Records import _conversation_turn_html
from trulens.dashboard.tabs.Records import _partition_feedback_scopes
from trulens.dashboard.tabs.Records import _split_conversation_turns


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
