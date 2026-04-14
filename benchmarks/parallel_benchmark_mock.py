"""
Benchmark: Parallel vs Sequential Run.start() and Run.compute_metrics()

Uses a simulated RAG app with artificial I/O latency (0.5s per LLM call) to
demonstrate parallelization speedup. Mocks the DAO layer so no Snowflake
connection is needed.

Each RAG invocation takes ~1s (0.5s retrieval + 0.5s generation).
8 rows sequential = ~8s, parallel (4 workers) = ~2s expected.
"""

import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
from trulens.apps.app import TruApp
from trulens.core import TruSession
from trulens.core.otel.instrument import instrument
from trulens.core.run import Run
from trulens.core.run import RunStatus
from trulens.otel.semconv.trace import SpanAttributes

SIMULATED_LATENCY = 0.5


class FakeRAG:
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list:
        time.sleep(SIMULATED_LATENCY)
        return [f"Context about {query[:30]}"]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, context_list: list) -> str:
        time.sleep(SIMULATED_LATENCY)
        return f"Answer for: {query[:50]}"

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        ctx = self.retrieve(query=query)
        return self.generate(query=query, context_list=ctx)


test_dataset = pd.DataFrame({
    "input": [
        "When was Yellowstone established?",
        "What are the famous rock formations in Yosemite?",
        "How deep is the Grand Canyon?",
        "What river runs through Zion Canyon?",
        "What wildlife can be found in Glacier National Park?",
        "Where is Acadia National Park located?",
        "Which national park has geothermal features?",
        "What is the Going-to-the-Sun Road?",
    ]
})


def make_run(label, invocation_workers=None, metric_workers=None):
    session = TruSession()
    session.reset_database()

    rag = FakeRAG()
    tru_app = TruApp(
        rag,
        app_name="Bench App",
        app_version=label,
        main_method=rag.query,
    )

    mock_dao = MagicMock()
    mock_dao.create_new_run.return_value = pd.DataFrame({
        "col": [
            '{"run_name": "bench_run", "object_name": "TEST", "object_type": "EXTERNAL AGENT", "run_metadata": {}, "source_info": {"name": "test", "column_spec": {}, "source_type": "TABLE"}}'
        ]
    })
    mock_dao.get_run_status.return_value = RunStatus.CREATED.value

    run = Run(
        run_dao=mock_dao,
        app=tru_app,
        main_method_name="query",
        tru_session=session,
        object_name="TEST",
        object_type="EXTERNAL AGENT",
        run_name="bench_run",
        run_metadata=Run.RunMetadata(),
        source_info=Run.SourceInfo(
            name="test", column_spec={"input": "input"}, source_type="DATAFRAME"
        ),
        invocation_max_workers=invocation_workers,
        metric_max_workers=metric_workers,
    )
    return run, session


def bench_start(run):
    with patch.object(Run, "get_status", return_value=RunStatus.CREATED.value):
        t0 = time.perf_counter()
        run.start(input_df=test_dataset)
        return time.perf_counter() - t0


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK: Parallel vs Sequential run.start()")
    print(f"Dataset: {len(test_dataset)} rows")
    print(f"Simulated latency: {SIMULATED_LATENCY}s per LLM call (2 per row)")
    print(
        f"Expected sequential: ~{len(test_dataset) * SIMULATED_LATENCY * 2:.0f}s"
    )
    print(
        f"Expected parallel (4 workers): ~{len(test_dataset) * SIMULATED_LATENCY * 2 / 4:.0f}s"
    )
    print("=" * 60)

    print("\n--- run.start() benchmark ---")

    run_seq, _ = make_run("sequential", invocation_workers=1)
    t_seq = bench_start(run_seq)
    print(f"  Sequential (workers=1): {t_seq:.2f}s")

    run_par, _ = make_run("parallel", invocation_workers=None)
    t_par = bench_start(run_par)
    print(f"  Parallel   (default) : {t_par:.2f}s")

    speedup = t_seq / t_par if t_par > 0 else float("inf")
    print(f"  Speedup: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  run.start(): {t_seq:.2f}s -> {t_par:.2f}s  ({speedup:.2f}x)")
    print("=" * 60)
