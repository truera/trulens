"""TruLens CI/CD evaluation gate.

Runs a tiny TruLens evaluation suite and exits non-zero when any feedback score
falls below a configurable threshold. The script is intentionally self-contained
so it can be copied into CI systems as a quality gate.

Environment variables:
    OPENAI_API_KEY      Required. Key for the OpenAI provider used as LLM judge.
    TRULENS_MIN_SCORE   Optional. Minimum acceptable score (0-1). Defaults to 0.7.
    TRULENS_EVAL_MODEL  Optional. Judge model. Defaults to "gpt-4o-mini".
    TRULENS_JUNIT_XML   Optional. Path to write a JUnit XML report.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from trulens.apps.app import TruApp
from trulens.core import Metric, Selector, TruSession
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

KNOWLEDGE_BASE = {
    "trulens": (
        "TruLens is an open-source library for evaluating and tracking the "
        "quality of LLM applications using feedback functions."
    ),
    "rag": (
        "Retrieval-augmented generation augments an LLM with documents "
        "retrieved from an external knowledge base before generating an answer."
    ),
    "feedback": (
        "A feedback function programmatically scores an LLM app's inputs and "
        "outputs, for example measuring relevance or groundedness."
    ),
}

TEST_SET = [
    ("What is TruLens?", "trulens"),
    ("How does RAG work?", "rag"),
    ("What does a feedback function do?", "feedback"),
]


class TinyRAG:
    """A minimal RAG app instrumented for TruLens evaluation."""

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str, key: str) -> list[str]:
        return [KNOWLEDGE_BASE[key]]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, contexts: list[str]) -> str:
        return contexts[0]

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def answer(self, query: str, key: str) -> str:
        contexts = self.retrieve(query, key)
        return self.generate(query, contexts)


def build_feedbacks():
    """Configure RAG triad feedback functions using the OpenAI provider."""
    from trulens.providers.openai import OpenAI

    provider = OpenAI(model_engine=os.environ.get("TRULENS_EVAL_MODEL", "gpt-4o-mini"))

    return [
        Metric(
            implementation=provider.relevance_with_cot_reasons,
            name="Answer Relevance",
            selectors={
                "prompt": Selector.select_record_input(),
                "response": Selector.select_record_output(),
            },
        ),
        Metric(
            implementation=provider.context_relevance_with_cot_reasons,
            name="Context Relevance",
            selectors={
                "question": Selector.select_record_input(),
                "context": Selector.select_context(collect_list=False),
            },
        ),
        Metric(
            implementation=provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "statement": Selector.select_record_output(),
            },
        ),
    ]


def write_junit(path: str, failures: list[tuple[str, float]], scores: dict[str, float], min_score: float) -> None:
    suite = ET.Element(
        "testsuite",
        name="trulens-eval-gate",
        tests=str(len(scores)),
        failures=str(len(failures)),
    )
    failed = {name: score for name, score in failures}
    for name, score in scores.items():
        case = ET.SubElement(suite, "testcase", classname="TruLensEvalGate", name=name)
        if name in failed:
            failure = ET.SubElement(case, "failure", message=f"{name} below threshold")
            failure.text = f"{name}: {score:.3f} < {min_score:.3f}"

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(output, encoding="utf-8", xml_declaration=True)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    min_score = float(os.environ.get("TRULENS_MIN_SCORE", "0.7"))

    session = TruSession()
    session.reset_database()

    app = TinyRAG()
    tru_app = TruApp(
        app,
        app_name="CI Eval Gate",
        app_version="v1",
        feedbacks=build_feedbacks(),
    )

    with tru_app as recording:
        for query, key in TEST_SET:
            app.answer(query, key)

    recording.retrieve_feedback_results(timeout=300)

    leaderboard = session.get_leaderboard()
    print("\nLeaderboard:\n", leaderboard.to_string())

    ignore = {"latency", "total_cost", "total_tokens", "Latency", "Cost (USD)"}
    feedback_cols = [
        col for col in leaderboard.columns
        if col not in ignore and leaderboard[col].dtype.kind in "fi"
    ]

    scores: dict[str, float] = {}
    failures: list[tuple[str, float]] = []
    for col in feedback_cols:
        score = float(leaderboard[col].mean())
        scores[col] = score
        status = "PASS" if score >= min_score else "FAIL"
        print(f"  {status}  {col}: {score:.3f} (min {min_score})")
        if score < min_score:
            failures.append((col, score))

    if junit_path := os.environ.get("TRULENS_JUNIT_XML"):
        write_junit(junit_path, failures, scores, min_score)
        print(f"\nWrote JUnit report to {junit_path}")

    if failures:
        print(f"\nEval gate FAILED: {len(failures)} metric(s) below {min_score}.", file=sys.stderr)
        return 1

    print(f"\nEval gate PASSED: all metrics >= {min_score}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
