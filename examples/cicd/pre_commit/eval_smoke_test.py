"""Fast TruLens eval smoke test for pre-commit.

This hook runs a tiny evaluation before code is committed. Keep it fast: one or
two cases, one lightweight feedback function, and a small judge model.

Environment variables:
    OPENAI_API_KEY      Required unless SKIP_TRULENS_PRECOMMIT=1.
    TRULENS_MIN_SCORE   Optional. Minimum acceptable score (0-1). Defaults to 0.7.
    TRULENS_EVAL_MODEL  Optional. Judge model. Defaults to "gpt-4o-mini".
"""

from __future__ import annotations

import os
import sys

from trulens.apps.app import TruApp
from trulens.core import Metric, Selector, TruSession
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes


class TinyQA:
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "question",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def answer(self, question: str) -> str:
        return "TruLens evaluates and tracks LLM application quality with feedback functions."


def main() -> int:
    if os.environ.get("SKIP_TRULENS_PRECOMMIT") == "1":
        print("Skipping TruLens pre-commit smoke test.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Set it or run with SKIP_TRULENS_PRECOMMIT=1.",
            file=sys.stderr,
        )
        return 1

    from trulens.providers.openai import OpenAI

    min_score = float(os.environ.get("TRULENS_MIN_SCORE", "0.7"))
    provider = OpenAI(model_engine=os.environ.get("TRULENS_EVAL_MODEL", "gpt-4o-mini"))

    f_relevance = Metric(
        implementation=provider.relevance,
        name="Answer Relevance",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    session = TruSession()
    session.reset_database()

    app = TinyQA()
    tru_app = TruApp(
        app,
        app_name="Pre-Commit Eval Smoke Test",
        app_version="v1",
        feedbacks=[f_relevance],
    )

    with tru_app as recording:
        app.answer("What is TruLens used for?")

    results = recording.retrieve_feedback_results(timeout=120)
    score = float(results["Answer Relevance"].mean())

    print(f"Answer Relevance: {score:.3f} (min {min_score})")
    if score < min_score:
        print("TruLens pre-commit smoke test failed.", file=sys.stderr)
        return 1

    print("TruLens pre-commit smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
