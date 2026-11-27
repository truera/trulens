"""TruLens CI/CD evaluation gate.

Runs a tiny TruLens evaluation suite and exits non-zero when any feedback score
falls below a configurable threshold or was not computed. The script is intentionally self-contained
so it can be copied into CI systems as a quality gate.

Environment variables:
    OPENAI_API_KEY      Required. Key for the OpenAI provider used as LLM judge.
    TRULENS_MIN_SCORE   Optional. Minimum acceptable score (0-1). Defaults to 0.7.
    TRULENS_EVAL_MODEL  Optional. Judge model. Defaults to "gpt-4o-mini".
    TRULENS_JUNIT_XML   Optional. Path to write a JUnit XML report.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pandas as pd
from trulens.apps.app import TruApp
from trulens.core import Metric
from trulens.core import Selector
from trulens.core import TruSession
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

    provider = OpenAI(
        model_engine=os.environ.get("TRULENS_EVAL_MODEL", "gpt-4o-mini")
    )

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


def check_scores(
    results: pd.DataFrame, metric_names: list[str], min_score: float
) -> list[tuple[str, float, str | None]]:
    """Check each configured metric against the threshold.

    The scores come from the per-record results, not the leaderboard. The
    leaderboard also has numeric columns that are not metrics, such as tool-call
    counts, and a metric whose judge failed on every record is not numeric there.

    Args:
        results: Per-record scores from `retrieve_feedback_results`, one column
            per metric.
        metric_names: Names of the metrics the app was recorded with.
        min_score: Lowest acceptable mean score.

    Returns:
        One `(metric, mean score, failure reason)` per metric. The reason is
        `None` when the metric passes. A record without a score fails the
        metric, because it was not evaluated.
    """
    checks = []
    for name in metric_names:
        if name in results.columns:
            scores = pd.to_numeric(results[name], errors="coerce")
        else:
            scores = pd.Series(float("nan"), index=results.index)
        mean = float(scores.mean())
        unscored = int(scores.isna().sum())
        if scores.empty:
            reason = "no records were evaluated"
        elif unscored:
            reason = f"no score for {unscored} of {len(scores)} record(s)"
        elif mean < min_score:
            reason = f"{mean:.3f} < {min_score:.3f}"
        else:
            reason = None
        checks.append((name, mean, reason))
    return checks


def write_junit(path: str, checks: list[tuple[str, float, str | None]]) -> None:
    failures = [name for name, _, reason in checks if reason]
    suite = ET.Element(
        "testsuite",
        name="trulens-eval-gate",
        tests=str(len(checks)),
        failures=str(len(failures)),
    )
    for name, mean, reason in checks:
        case = ET.SubElement(
            suite, "testcase", classname="TruLensEvalGate", name=name
        )
        if reason:
            failure = ET.SubElement(
                case, "failure", message=f"{name} failed the gate"
            )
            failure.text = f"{name}: {reason}"

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
    feedbacks = build_feedbacks()
    tru_app = TruApp(
        app,
        app_name="CI Eval Gate",
        app_version="v1",
        feedbacks=feedbacks,
    )

    with tru_app as recording:
        for query, key in TEST_SET:
            app.answer(query, key)

    results = recording.retrieve_feedback_results(timeout=300)

    leaderboard = session.get_leaderboard()
    print("\nLeaderboard:\n", leaderboard.to_string())

    checks = check_scores(
        results, [feedback.name for feedback in feedbacks], min_score
    )
    failures = [(name, reason) for name, _, reason in checks if reason]
    for name, mean, reason in checks:
        if reason:
            print(f"  FAIL  {name}: {reason}")
        else:
            print(f"  PASS  {name}: {mean:.3f} (min {min_score})")

    if junit_path := os.environ.get("TRULENS_JUNIT_XML"):
        write_junit(junit_path, checks)
        print(f"\nWrote JUnit report to {junit_path}")

    if failures:
        print(
            f"\nEval gate FAILED: {len(failures)} metric(s) below {min_score} "
            "or not scored.",
            file=sys.stderr,
        )
        return 1

    print(f"\nEval gate PASSED: all metrics >= {min_score}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
