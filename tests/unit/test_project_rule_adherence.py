"""Tests for coding-agent project-rule adherence feedback."""

import json
from unittest import mock

import pytest
from trulens.feedback import llm_provider
from trulens.feedback import output_schemas
from trulens.feedback import templates


class _ResponseEndpoint:
    def __init__(self, response):
        self.response = response
        self.call = None

    def run_in_pace(self, **kwargs):
        self.call = kwargs
        return self.response


def _provider_with_response(response):
    provider = mock.create_autospec(llm_provider.LLMProvider, instance=True)
    provider.endpoint = _ResponseEndpoint(response)
    provider._is_reasoning_model.return_value = False
    provider._build_criteria_with_instructions.side_effect = (
        lambda *, criteria, default_criteria, additional_instructions: (
            default_criteria
        )
    )
    return provider


def test_project_rule_adherence_scores_only_decided_rules() -> None:
    response = output_schemas.ProjectRuleAdherenceResponse(
        evaluations=[
            output_schemas.ProjectRuleEvaluation(
                rule="Run lint before finishing.",
                verdict="violated",
                evidence="No executed lint command appears in the trace.",
                span_ids=[],
                determination="judge",
            ),
            output_schemas.ProjectRuleEvaluation(
                rule="Do not edit secrets/**.",
                verdict="not_applicable",
                evidence=(
                    "The only proposed secrets edit was blocked before "
                    "execution."
                ),
                span_ids=["blocked-span"],
                determination="judge",
            ),
        ]
    )
    provider = _provider_with_response(response)
    trace = json.dumps({
        "spans": [
            {
                "span_id": "edit-span",
                "trulens.coding_agent.native_event": "PostToolUse",
                "trulens.coding_agent.diff": "--- a/src/app.py\n+++ b/src/app.py",
            },
            {
                "span_id": "blocked-span",
                "trulens.coding_agent.native_event": "PreToolUse",
                "gen_ai.tool.call.arguments": {
                    "path": "secrets/token.txt",
                    "content": "would-be edit",
                },
            },
        ]
    })

    score, reason = (
        llm_provider.LLMProvider.project_rule_adherence_with_cot_reasons(
            provider,
            rules="Run lint before finishing.\nDo not edit secrets/**.",
            trace=trace,
        )
    )

    assert score == 0.0
    assert reason["followed_count"] == 0
    assert reason["violated_count"] == 1
    assert reason["not_applicable_count"] == 1
    assert reason["scored_rule_count"] == 1
    assert reason["excluded_unexecuted_call_count"] == 1
    assert reason["excluded_unexecuted_span_ids"] == ["blocked-span"]
    assert reason["rules"][0]["determination"] == "judge"


def test_project_rule_adherence_passes_deterministic_checks_to_judge() -> None:
    deterministic_rule = "Run required command containing 'ruff check'."
    response = output_schemas.ProjectRuleAdherenceResponse(
        evaluations=[
            output_schemas.ProjectRuleEvaluation(
                rule=deterministic_rule,
                verdict="followed",
                evidence="The required command ran in spans ['lint-span'].",
                span_ids=["lint-span"],
                determination="deterministic",
            )
        ]
    )
    provider = _provider_with_response(response)
    trace = json.dumps({
        "spans": [
            {
                "span_id": "lint-span",
                "trulens.coding_agent.native_event": "PostToolUse",
                "gen_ai.tool.call.arguments": {
                    "command": "poetry run ruff check ."
                },
                "gen_ai.tool.call.result": "All checks passed!",
            }
        ]
    })

    score, reason = (
        llm_provider.LLMProvider.project_rule_adherence_with_cot_reasons(
            provider,
            rules="Run the repository lint command.",
            trace=trace,
            required_commands=["ruff check"],
        )
    )

    assert score == 1.0
    assert reason["rules"][0]["determination"] == "deterministic"
    evaluation_data = json.loads(
        provider.endpoint.call["messages"][1]["content"].split("\n", 2)[2]
    )
    assert evaluation_data["deterministic_results"] == [
        {
            "rule": deterministic_rule,
            "verdict": "followed",
            "evidence": "The required command ran in spans ['lint-span'].",
            "span_ids": ["lint-span"],
            "determination": "deterministic",
        }
    ]


def test_project_rule_adherence_read_is_not_an_edit() -> None:
    deterministic_rule = "Do not edit paths matching 'secrets/**'."
    response = output_schemas.ProjectRuleAdherenceResponse(
        evaluations=[
            output_schemas.ProjectRuleEvaluation(
                rule=deterministic_rule,
                verdict="followed",
                evidence="No executed edit matched 'secrets/**'.",
                span_ids=[],
                determination="deterministic",
            )
        ]
    )
    provider = _provider_with_response(response)
    trace = json.dumps({
        "spans": [
            {
                "span_id": "read-span",
                "trulens.coding_agent.native_event": "PostToolUse",
                "gen_ai.tool.name": "Read",
                "gen_ai.tool.call.arguments": {
                    "file_path": "secrets/token.txt"
                },
                "gen_ai.tool.call.result": "redacted",
            }
        ]
    })

    score, reason = (
        llm_provider.LLMProvider.project_rule_adherence_with_cot_reasons(
            provider,
            rules="Do not edit secrets/**.",
            trace=trace,
            forbidden_path_globs=["secrets/**"],
        )
    )

    assert score == 1.0
    assert reason["rules"][0]["verdict"] == "followed"
    assert reason["rules"][0]["span_ids"] == []


def test_project_rule_adherence_ignores_commands_in_tool_results() -> None:
    deterministic_rule = "Run required command containing 'ruff check'."
    response = output_schemas.ProjectRuleAdherenceResponse(
        evaluations=[
            output_schemas.ProjectRuleEvaluation(
                rule=deterministic_rule,
                verdict="violated",
                evidence="No executed tool span contains the required command.",
                span_ids=[],
                determination="deterministic",
            )
        ]
    )
    provider = _provider_with_response(response)
    trace = json.dumps({
        "spans": [
            {
                "span_id": "grep-span",
                "trulens.coding_agent.native_event": "PostToolUse",
                "gen_ai.tool.name": "Grep",
                "gen_ai.tool.call.arguments": {"pattern": "lint command"},
                "gen_ai.tool.call.result": "Run poetry ruff check before PR.",
            }
        ]
    })

    score, reason = (
        llm_provider.LLMProvider.project_rule_adherence_with_cot_reasons(
            provider,
            rules="Run the repository lint command.",
            trace=trace,
            required_commands=["ruff check"],
        )
    )

    assert score == 0.0
    assert reason["rules"][0]["verdict"] == "violated"
    assert reason["rules"][0]["span_ids"] == []


def test_project_rule_adherence_requires_an_applicable_rule() -> None:
    response = output_schemas.ProjectRuleAdherenceResponse(
        evaluations=[
            output_schemas.ProjectRuleEvaluation(
                rule="Use Java for Java files.",
                verdict="not_applicable",
                evidence="No Java files were involved.",
                span_ids=[],
                determination="judge",
            )
        ]
    )
    provider = _provider_with_response(response)

    with pytest.raises(ValueError, match="No applicable project rules"):
        llm_provider.LLMProvider.project_rule_adherence_with_cot_reasons(
            provider,
            rules="Use Java for Java files.",
            trace="{}",
        )


def test_project_rule_adherence_template_is_exported() -> None:
    assert "ProjectRuleAdherence" in templates.__all__
    assert templates.ProjectRuleAdherence is not None
