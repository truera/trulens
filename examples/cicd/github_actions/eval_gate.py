"""TruLens CI/CD evaluation gate.

Runs a tiny TruLens evaluation over a fixed test set and exits non-zero if any
feedback score falls below a configurable threshold. Designed to be dropped into
a CI pipeline (GitHub Actions, GitLab CI, etc.) as a quality gate.

Usage:
    OPENAI_API_KEY=sk-...  TRULENS_MIN_SCORE=0.7  python eval_gate.py

Environment variables:
    OPENAI_API_KEY     Required. Key for the OpenAI provider used as LLM judge.
    TRULENS_MIN_SCORE  Optional. Minimum acceptable score (0-1). Defaults to 0.7.
    TRULENS_EVAL_MODEL Optional. Judge model. Defaults to "gpt-4o-mini".
"""

from __future__ import annotations

import os
import sys

from trulens.apps.app import TruApp
from trulens.core import Metric, Selector, TruSession
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

# A tiny, self-contained "knowledge base" so the example runs in seconds with no
# external services beyond the LLM judge.
KNOWLEDGE_BASE = {
    "trulens": (
        "TruLens is an open-source library for evaluating and tracking the "
        "quality of LLM applications using feedback functions."
    ),
    "rag": (
        "Retrieval-augmented generation (RAG) augments an LLM with documents "
        "retrieved from an external knowledge base before generating an answer."
    ),
    "feedback": (
        "A feedback function programmatically scores an LLM app's inputs and "
        "outputs, for example measuring relevance or groundedness."
    ),
}

# Test set: question -> retrieval key. Keep this small (3-5) so the gate runs fast.
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
    """Configure the RAG triad feedback functions using the OpenAI provider."""
    from trulens.providers.openai import OpenAI

    provider = OpenAI(model_engine=os.environ.get("TRULENS_EVAL_MODEL", "gpt-4o-mini"))

    f_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="Answer Relevance",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )
    f_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="Context Relevance",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
    )
    f_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="Groundedness",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
        },
    )
    return [f_answer_relevance, f_context_relevance, f_groundedness]


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

    # Block until all feedback evaluations have completed.
    recording.retrieve_feedback_results(timeout=300)

    leaderboard = session.get_leaderboard()
    print("\nLeaderboard:\n", leaderboard.to_string())

    # Compare every feedback column against the threshold. Non-feedback columns
    # (latency, cost) are ignored.
    ignore = {"latency", "total_cost", "total_tokens", "Latency", "Cost (USD)"}
    feedback_cols = [
        c for c in leaderboard.columns
        if c not in ignore and leaderboard[c].dtype.kind in "fi"
    ]

    failures = []
    for col in feedback_cols:
        score = float(leaderboard[col].mean())
        status = "PASS" if score >= min_score else "FAIL"
        print(f"  {status}  {col}: {score:.3f} (min {min_score})")
        if score < min_score:
            failures.append((col, score))

    if failures:
        print(
            f"\nEval gate FAILED: {len(failures)} metric(s) below {min_score}.",
            file=sys.stderr,
        )
        return 1

    print(f"\nEval gate PASSED: all metrics >= {min_score}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
