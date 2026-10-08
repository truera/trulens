"""Offline checks for the notebook's two protocol adapters."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import httpx
import pytest
from trulens.core import Metric
from trulens.feedback import Jury

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/expositional/models/local_and_OSS_models/clef_judge_comparison"
)
spec = importlib.util.spec_from_file_location(
    "decision_judges", EXAMPLE / "decision_judges.py"
)
judges = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = judges
spec.loader.exec_module(judges)


@pytest.mark.parametrize("value,expected", [(0, 0), (2.4, 0.6), (4, 1)])
def test_clef_typed_score(value, expected):
    def respond(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/systemone"
        assert payload["state"] == {"source": "Article", "summary": "Summary"}
        assert payload["questions"]["consistency"]["criteria"] == judges.LEVELS
        return httpx.Response(
            200, json={"answers": {"consistency": {"score": value}}}
        )

    with httpx.Client(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(respond),
    ) as client:
        judge = judges.Clef(client)
        assert judge.score("Article", "Summary") == expected
        assert judge.calls[0]["score"] == expected
        assert judge.calls[0]["status"] == "ok"
        assert judge.calls[0]["seconds"] >= 0


@pytest.mark.parametrize("value", [True, "3", -1, 5, None, float("nan")])
def test_clef_rejects_invalid_scores(value):
    with httpx.Client(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=json.dumps({
                    "answers": {"consistency": {"score": value}}
                }),
            )
        ),
    ) as client:
        judge = judges.Clef(client)
        with pytest.raises(ValueError):
            judge.score("Article", "Summary")
        assert judge.calls[0]["status"] == "error"
        assert "score" not in judge.calls[0]


def cortex_client(text='{"score": 4}', status=200, stop="end_turn"):
    def respond(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/v2/cortex/v1/messages"
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "medium"}
        assert "temperature" not in body
        assert "response_format" not in body
        return httpx.Response(
            status,
            json={
                "stop_reason": stop,
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 100, "output_tokens": 5},
            },
        )

    return httpx.Client(
        base_url="https://example.snowflakecomputing.com",
        transport=httpx.MockTransport(respond),
    )


def cortex_judge(client):
    connection = types.SimpleNamespace(
        rest=types.SimpleNamespace(token="test-token-not-a-secret")
    )
    return judges.CortexMessages("claude-sonnet-5-5", connection, client)


def test_native_score_parsing_and_serialization():
    with cortex_client() as client:
        judge = cortex_judge(client)
        assert judge.score("Article", "Summary") == 0.75
        assert judge.calls[0]["score"] == 0.75
        assert judge.calls[0]["usage"]["input_tokens"] == 100
        serialized = judge.model_dump()
        assert not {"connection", "client", "calls"} & serialized.keys()
        assert "test-token-not-a-secret" not in str(serialized)


@pytest.mark.parametrize(
    "text", ['{"score": 6}', '{"score": "bad"}', "no rating"]
)
def test_native_parse_failures_are_not_successes(text):
    with cortex_client(text=text) as client:
        judge = cortex_judge(client)
        with pytest.raises(Exception):
            judge.score("Article", "Summary")
        assert judge.calls[0]["status"] == "error"
        assert "score" not in judge.calls[0]


@pytest.mark.parametrize(
    "status,stop", [(400, "end_turn"), (200, "max_tokens")]
)
def test_transport_failures_have_no_retries(status, stop):
    with cortex_client(status=status, stop=stop) as client:
        judge = cortex_judge(client)
        with pytest.raises(Exception):
            judge.score("Article", "Summary")
        assert len(judge.calls) == 1
        assert judge.calls[0]["status"] == "error"


def test_metric_native_repeated_jury():
    with cortex_client() as client:
        judge = cortex_judge(client)
        metric = Metric(
            implementation=Jury.repeated(
                judge,
                method="score",
                n_trials=3,
                threshold=0.75,
                max_workers=1,
            ),
            name="Consistency",
        )
        score, metadata = metric(source="Article", summary="Summary")
        assert score == 0.75
        assert metadata["reliability.n_scores"] == 3
        assert metadata["reliability.flip_rate"] == 0
        assert metadata["reliability.score_std"] == 0
        assert len(judge.calls) == 3


def test_published_snapshot_is_complete_and_sanitized():
    payload = json.loads((EXAMPLE / "results_snapshot.json").read_text())
    metadata = payload["metadata"]
    rows = [
        dict(zip(payload["attempts"]["columns"], values, strict=True))
        for values in payload["attempts"]["data"]
    ]
    measured = [row for row in rows if row["phase"] == "measurement"]
    assert len(measured) == metadata["scheduled_judgments"] == 900
    keys = {(row["example_id"], row["name"], row["repeat"]) for row in measured}
    assert len(keys) == 900
    assert len({row["example_id"] for row in measured}) == 100
    assert {row["repeat"] for row in measured} == {0, 1, 2}
    for row in measured:
        if row["status"] == "ok":
            assert 0 <= row["score"] <= 1
        else:
            assert row["score"] is None
    text = json.dumps(payload).lower()
    for forbidden in [
        "authorization",
        "snowhouse",
        "jreini",
        "request_id",
        "api_key",
    ]:
        assert forbidden not in text


def test_saved_notebook_outputs_describe_published_run():
    payload = json.loads((EXAMPLE / "results_snapshot.json").read_text())
    notebook = json.loads((EXAMPLE / "clef_judge_comparison.ipynb").read_text())
    outputs = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    ]
    streams = "".join(
        "".join(output.get("text", []))
        for output in outputs
        if output["output_type"] == "stream"
    )
    metadata = payload["metadata"]
    assert f"Run: {metadata['run_started_utc']}" in streams
    assert (
        f"{metadata['examples']} summaries; {metadata['repeats']} trials; "
        f"{metadata['scheduled_judgments']} scheduled judgments"
    ) in streams
    assert all(output["output_type"] != "error" for output in outputs)
    assert any("image/png" in output.get("data", {}) for output in outputs)
    assert "/Users/" not in json.dumps(outputs)
