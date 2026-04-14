"""
Benchmark: Parallel vs Sequential Run.start() with Snowflake connection.

Uses simulated I/O latency (0.5s per LLM call) so no OpenAI key is needed.
Connects to Snowflake via connection name to use real DAO layer.

Usage:
  SNOWFLAKE_CONNECTION_NAME=DEVREL_ENTERPRISE \
    python benchmarks/run_benchmark_sf.py
"""

import os
import time
import uuid

import numpy as np
import pandas as pd
from snowflake.snowpark import Session
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import Metric
from trulens.core import Selector
from trulens.core import TruSession
from trulens.core.otel.instrument import instrument
from trulens.core.run import RunConfig
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


def fake_groundedness(source, statement, question) -> float:
    time.sleep(SIMULATED_LATENCY)
    return 0.9


def fake_relevance(prompt, response) -> float:
    time.sleep(SIMULATED_LATENCY)
    return 0.85


def fake_context_relevance(question, context) -> float:
    time.sleep(SIMULATED_LATENCY)
    return 0.88


f_groundedness = Metric(
    implementation=fake_groundedness,
    name="Groundedness",
    selectors={
        "source": Selector.select_context(collect_list=True),
        "statement": Selector.select_record_output(),
        "question": Selector.select_record_input(),
    },
)
f_answer_relevance = Metric(
    implementation=fake_relevance,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)
f_context_relevance = Metric(
    implementation=fake_context_relevance,
    name="Context Relevance",
    selectors={
        "question": Selector.select_record_input(),
        "context": Selector.select_context(collect_list=False),
    },
    agg=np.mean,
)

METRICS = [f_groundedness, f_answer_relevance, f_context_relevance]


def bench_start(run):
    t0 = time.perf_counter()
    run.start(input_df=test_dataset)
    return time.perf_counter() - t0


def bench_metrics(run):
    t0 = time.perf_counter()
    run.compute_metrics(METRICS)
    return time.perf_counter() - t0


if __name__ == "__main__":
    conn_name = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "DEVREL_ENTERPRISE")
    print(f"Connecting to Snowflake (connection={conn_name})...")
    snowpark_session = (
        Session.builder.config("connection_name", conn_name)
        .config("database", "TRULENS_TEST")
        .config("schema", "PUBLIC")
        .config("warehouse", "COMPUTE")
        .create()
    )
    connector = SnowflakeConnector(snowpark_session=snowpark_session)
    session = TruSession(connector=connector)
    print("Connected.")

    print("=" * 60)
    print("BENCHMARK: Parallel vs Sequential (Snowflake connection)")
    print(f"Dataset: {len(test_dataset)} rows, Metrics: {len(METRICS)}")
    print(f"Simulated latency: {SIMULATED_LATENCY}s per call (2 per row)")
    print("=" * 60)

    # --- run.start() benchmark ---
    print("\n--- run.start() benchmark ---")

    uid = uuid.uuid4().hex[:6]
    rag_seq = FakeRAG()
    tru_seq = TruApp(
        rag_seq,
        connector=connector,
        app_name="Bench",
        app_version="seq_invoke",
        main_method=rag_seq.query,
        feedbacks=[],
        start_evaluator=False,
    )
    print("  Created sequential TruApp")
    run_seq = tru_seq.add_run(
        RunConfig(
            run_name=f"bench_seq_{uid}",
            dataset_name="test",
            source_type="DATAFRAME",
            dataset_spec={"input": "input"},
            invocation_max_workers=1,
        )
    )
    print("  Created sequential run, starting...")
    t_seq = bench_start(run_seq)
    print(f"  Sequential (workers=1): {t_seq:.2f}s")

    uid = uuid.uuid4().hex[:6]
    rag_par = FakeRAG()
    tru_par = TruApp(
        rag_par,
        connector=connector,
        app_name="Bench",
        app_version="par_invoke",
        main_method=rag_par.query,
        feedbacks=[],
        start_evaluator=False,
    )
    print("  Created parallel TruApp")
    run_par = tru_par.add_run(
        RunConfig(
            run_name=f"bench_par_{uid}",
            dataset_name="test",
            source_type="DATAFRAME",
            dataset_spec={"input": "input"},
        )
    )
    print("  Created parallel run, starting...")
    t_par = bench_start(run_par)
    print(f"  Parallel   (default) : {t_par:.2f}s")

    speedup_start = t_seq / t_par if t_par > 0 else float("inf")
    print(f"  Speedup: {speedup_start:.2f}x")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (Snowflake)")
    print(
        f"  run.start(): {t_seq:.2f}s -> {t_par:.2f}s  ({speedup_start:.2f}x)"
    )
    print("=" * 60)
