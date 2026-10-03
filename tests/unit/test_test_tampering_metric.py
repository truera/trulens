import json
from typing import Any, Dict, Optional, Sequence, Tuple, Type

from opentelemetry import trace as otel_trace
import pandas as pd
from pydantic import BaseModel
import pytest
from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.feedback.selector import Trace
from trulens.core.metric.metric import Metric
from trulens.core.metric.metric import SkipEval
from trulens.feedback.computer import _call_feedback_function
from trulens.feedback.computer import _call_feedback_function_under_eval_span
from trulens.feedback.llm_provider import LLMProvider
from trulens.otel.semconv.trace import SpanAttributes


class MockLLMProvider(LLMProvider):
    model_config = {"extra": "allow"}
    judge_result: Tuple[float, Dict] = (1.0, {"reason": "Test reason"})
    judge_calls: int = 0
    last_system_prompt: Optional[str] = None
    last_user_prompt: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(endpoint=None, model_engine="mock-model", **kwargs)

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict[Any, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> str:
        raise AssertionError("The judge should not be called")

    def generate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        min_score_val: int = 0,
        max_score_val: int = 3,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        self.judge_calls += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.judge_result


def test_no_test_file_diffs_skip_without_calling_judge():
    provider = MockLLMProvider()

    with pytest.raises(SkipEval, match="No captured diffs matched"):
        provider.test_tampering_with_cot_reasons(Trace())

    assert provider.judge_calls == 0


def _trace_with_diffs(*diffs: Any) -> Trace:
    trace = Trace()
    trace.events = pd.DataFrame([
        {
            "record_attributes": {
                SpanAttributes.CODING_AGENT.DIFF: json.dumps(diff),
            }
        }
        for diff in diffs
    ])
    return trace


def test_test_file_diffs_and_source_diffs_reach_judge():
    provider = MockLLMProvider()
    test_diff = {
        "file_path": "tests/test_agent.py",
        "edits": [{"old_text": "assert score", "new_text": "assert True"}],
    }
    source_diff = {
        "file_path": "src/agent.py",
        "edits": [{"old_text": "return score", "new_text": "return 1"}],
    }

    score, reasons = provider.test_tampering_with_cot_reasons(
        _trace_with_diffs(test_diff, source_diff)
    )

    assert score == provider.judge_result[0]
    assert reasons == provider.judge_result[1]
    assert provider.judge_calls == 1
    assert provider.last_system_prompt is not None
    assert provider.last_user_prompt is not None
    assert "weakens verification" in provider.last_system_prompt
    assert "tests/test_agent.py" in provider.last_user_prompt
    assert "src/agent.py" in provider.last_user_prompt
    assert "legitimate or weakening" in provider.last_system_prompt


@pytest.mark.parametrize(
    "file_path",
    [
        "src/test_agent.py",
        "src/agent_test.py",
        "tests/agent.py",
        "src/agent/__tests__/agent.js",
        "src/agent/agent.spec.ts",
        "src/agent/agent.test.js",
        "conftest.py",
    ],
)
def test_default_globs_select_test_files(file_path: str):
    provider = MockLLMProvider()

    provider.test_tampering_with_cot_reasons(
        _trace_with_diffs({"file_path": file_path, "edits": []})
    )

    assert provider.judge_calls == 1


def test_unified_diff_headers_identify_test_file_paths():
    provider = MockLLMProvider()
    diff = (
        "diff --git a/tests/test_agent.py b/tests/test_agent.py\n+assert True"
    )

    provider.test_tampering_with_cot_reasons(_trace_with_diffs(diff))

    assert provider.judge_calls == 1


def test_judge_weakening_verdict_is_returned():
    provider = MockLLMProvider(
        judge_result=(0.0, {"reason": "An assertion was replaced with True."})
    )

    score, reasons = provider.test_tampering_with_cot_reasons(
        _trace_with_diffs({
            "file_path": "tests/test_agent.py",
            "edits": [{"old_text": "assert score", "new_text": "assert True"}],
        })
    )

    assert score == 0.0
    assert reasons == {"reason": "An assertion was replaced with True."}


def test_malformed_trace_attributes_raise_instead_of_skipping():
    provider = MockLLMProvider()
    trace = Trace()
    trace.events = pd.DataFrame([{"record_attributes": None}])

    with pytest.raises(ValueError, match="record attributes must be a mapping"):
        provider.test_tampering_with_cot_reasons(trace)

    assert provider.judge_calls == 0


def test_empty_diff_file_path_raises_instead_of_skipping():
    provider = MockLLMProvider()

    with pytest.raises(ValueError, match="non-empty file path"):
        provider.test_tampering_with_cot_reasons(
            _trace_with_diffs({"file_path": "", "edits": []})
        )

    assert provider.judge_calls == 0


def test_custom_test_file_globs_replace_defaults():
    provider = MockLLMProvider()
    trace = _trace_with_diffs({"file_path": "tests/test_agent.py", "edits": []})

    with pytest.raises(SkipEval, match="No captured diffs matched"):
        provider.test_tampering_with_cot_reasons(
            trace, test_file_globs=["specs/**"]
        )

    assert provider.judge_calls == 0


def test_otel_eval_span_preserves_skip_eval_signal():
    def skip_metric():
        raise SkipEval("test skip")

    metric = Metric(implementation=skip_metric)
    root_span = otel_trace.get_tracer(__name__).start_span("eval-root")
    try:
        result = _call_feedback_function_under_eval_span(
            metric,
            {},
            root_span,
            is_only_child=True,
            eval_child_idx=0,
        )
    finally:
        root_span.end()

    from trulens.feedback import computer

    assert isinstance(result, computer._SkippedEvaluation)
    assert result.reason == "Metric evaluation skipped because test skip"


def test_otel_computer_does_not_treat_raw_none_as_skip():
    metric = Metric(implementation=lambda: None)
    root_span = otel_trace.get_tracer(__name__).start_span("eval-root")
    try:
        with pytest.raises(ValueError, match="raise SkipEval"):
            _call_feedback_function_under_eval_span(
                metric,
                {},
                root_span,
                is_only_child=True,
                eval_child_idx=0,
            )
    finally:
        root_span.end()


def test_otel_computer_aggregates_only_non_skipped_evaluations():
    def score_value(value: int) -> float:
        if value == 2:
            raise SkipEval("not applicable")
        return float(value)

    metric = Metric(implementation=score_value)
    result = _call_feedback_function(
        "test metric",
        metric,
        True,
        None,
        {"value": FeedbackFunctionInput(value=[1, 2, 3], collect_list=False)},
        "app",
        "version",
        "app-id",
        "run",
        "input-id",
        "record-id",
    )

    assert result == 2.0

    def skip_value(value: int) -> float:
        raise SkipEval(f"not applicable: {value}")

    skipped_metric = Metric(implementation=skip_value)
    skipped_result = _call_feedback_function(
        "test metric",
        skipped_metric,
        True,
        None,
        {"value": FeedbackFunctionInput(value=[1, 2], collect_list=False)},
        "app",
        "version",
        "app-id",
        "run",
        "input-id",
        "record-id",
    )

    assert skipped_result is None
