"""Unit tests for the TypeSafe Jev provider."""

from types import SimpleNamespace
from unittest import mock

import pytest
from trulens.providers.typesafe import provider as typesafe_provider
from typesafe_sdk import Choice
from typesafe_sdk import Noul
from typesafe_sdk import Score


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _provider(response):
    return typesafe_provider.TypeSafe(
        model_engine="jev-test",
        client=_FakeClient(response),
    )


def test_evaluate_batches_questions_and_preserves_provenance() -> None:
    response = SimpleNamespace(
        model="jev-test",
        answers={
            "supported": SimpleNamespace(type="noul", noul=0.8),
            "category": SimpleNamespace(
                type="choice",
                choice="billing",
                probabilities={"billing": 0.75, "technical": 0.25},
                confidence=0.5,
            ),
            "severity": SimpleNamespace(
                type="score",
                score=1.5,
                legend={"0": "low", "1": "medium", "2": "high"},
                probabilities={"0": 0.1, "1": 0.3, "2": 0.6},
                confidence=0.4,
            ),
        },
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )
    provider = _provider(response)
    state = {"document": "A duplicate charge appears on the invoice."}
    questions = {
        "supported": Noul(instructions="Is the charge duplicated?"),
        "category": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": None, "technical": None},
        ),
        "severity": Score(
            instructions="How severe is the issue?",
            criteria=["low", "medium", "high"],
        ),
    }

    decisions = provider.evaluate(state=state, questions=questions)

    assert len(provider.client.calls) == 1
    assert provider.client.calls[0]["questions"] == questions
    assert provider.client.calls[0]["model"] == "jev-test"
    assert decisions["supported"].value == pytest.approx(0.8)
    assert decisions["supported"].probabilities == {
        "no": pytest.approx(0.2),
        "yes": pytest.approx(0.8),
    }
    assert decisions["category"].value == "billing"
    assert decisions["severity"].value == pytest.approx(1.5)
    assert decisions["severity"].ordered_criteria == [
        ("0", "low"),
        ("1", "medium"),
        ("2", "high"),
    ]
    assert decisions["severity"].model == "jev-test"
    assert decisions["severity"].primitive == "score"
    assert decisions["severity"].input_tokens == 100
    assert decisions["severity"].output_tokens == 20
    assert len(decisions["severity"].question_template_hash) == 64
    assert len(decisions["severity"].state_hash) == 64
    assert decisions["severity"].raw_answer["legend"] == {
        "0": "low",
        "1": "medium",
        "2": "high",
    }


def test_score_normalizes_by_highest_level() -> None:
    provider = _provider(
        SimpleNamespace(
            model="jev-test",
            answers={
                "result": SimpleNamespace(
                    type="score",
                    score=1.5,
                    legend={"0": "low", "1": "medium", "2": "high"},
                    probabilities={"0": 0.0, "1": 0.5, "2": 0.5},
                    confidence=0.5,
                )
            },
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
    )

    assert provider.score(
        state="A state",
        instructions="How strong is it?",
        criteria=["low", "medium", "high"],
    ) == pytest.approx(0.75)


def test_groundedness_batches_all_statements() -> None:
    provider = _provider(
        SimpleNamespace(
            model="jev-test",
            answers={
                "statement_0": SimpleNamespace(type="noul", noul=0.9),
                "statement_1": SimpleNamespace(type="noul", noul=0.3),
            },
            usage=SimpleNamespace(input_tokens=20, output_tokens=2),
        )
    )

    score, metadata = provider.groundedness_measure(
        source="Paris is in France.",
        statements=["Paris is in France.", "Paris is in Germany."],
    )

    assert score == pytest.approx(0.6)
    assert len(provider.client.calls) == 1
    assert set(provider.client.calls[0]["questions"]) == {
        "statement_0",
        "statement_1",
    }
    assert set(metadata["statements"]) == {
        "Paris is in France.",
        "Paris is in Germany.",
    }


@pytest.mark.parametrize(
    "method_name",
    [
        "context_relevance_with_cot_reasons",
        "groundedness_measure_with_cot_reasons",
    ],
)
def test_reason_generating_methods_are_explicitly_unsupported(
    method_name: str,
) -> None:
    provider = _provider(SimpleNamespace())

    with pytest.raises(NotImplementedError, match="non-generative"):
        getattr(provider, method_name)()


def test_default_model_is_pinned() -> None:
    fake_client = mock.Mock()

    provider = typesafe_provider.TypeSafe(client=fake_client)

    assert provider.model_engine == "jev-1.13.0"


def test_endpoint_tracks_typed_decision_usage() -> None:
    response = SimpleNamespace(
        answers={
            "first": SimpleNamespace(type="noul", noul=0.8),
            "second": SimpleNamespace(type="noul", noul=0.2),
        },
        usage=SimpleNamespace(input_tokens=12, output_tokens=2),
    )
    provider = _provider(response)

    provider.evaluate(
        state="state",
        questions={
            "first": Noul(instructions="First?"),
            "second": Noul(instructions="Second?"),
        },
    )

    cost = provider.endpoint.global_callback.cost
    assert cost.n_requests == 1
    assert cost.n_successful_requests == 1
    assert cost.n_classification_requests == 1
    assert cost.n_classes == 2
    assert cost.n_prompt_tokens == 12
    assert cost.n_completion_tokens == 2
    assert cost.n_tokens == 14
