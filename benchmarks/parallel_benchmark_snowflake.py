"""
Benchmark: Parallel vs Sequential Run APIs on Snowflake

Same benchmark as parallel_benchmark_mock.py but using a real Snowflake
connection. Requires:
  - OPENAI_API_KEY set in environment
  - A Snowflake connection configured (snowpark session)

Usage:
  python benchmarks/parallel_benchmark_snowflake.py

Note: The parallelization code path is identical for OSS and Snowflake.
The mock benchmark (parallel_benchmark_mock.py) provides representative results.
The main Snowflake-specific difference is:
  - metric_max_workers only affects client-side Metric objects
  - Server-side string metrics are parallelized by Snowflake automatically
"""

import os
import time

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import numpy as np
from openai import OpenAI
import pandas as pd
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import Metric
from trulens.core import Selector
from trulens.core import TruSession
from trulens.core.otel.instrument import instrument
from trulens.core.run import RunConfig
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import OpenAI as fOpenAI

# --- Knowledge Base ---

docs = {
    "yellowstone": "Yellowstone National Park, established in 1872, was the world's first national park. Located primarily in Wyoming.",
    "yosemite": "Yosemite National Park is located in California's Sierra Nevada mountains. Known for its granite cliffs.",
    "grand_canyon": "The Grand Canyon, carved by the Colorado River, is 277 miles long, up to 18 miles wide, and over a mile deep.",
    "zion": "Zion National Park is located in southwestern Utah. The Virgin River runs through the canyon.",
    "glacier": "Glacier National Park in Montana contains over 700 miles of hiking trails and the Going-to-the-Sun Road.",
    "acadia": "Acadia National Park, located on Mount Desert Island in Maine, protects the highest rocky headlands along the Atlantic coastline.",
}

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)
chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection(
    name="national_parks_sf_bench", embedding_function=embedding_function
)
for k, v in docs.items():
    vector_store.add(k, documents=v)

oai_client = OpenAI()


class RAG:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list:
        results = vector_store.query(query_texts=query, n_results=3)
        return [doc for sublist in results["documents"] for doc in sublist]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context_list: list) -> str:
        if not context_list:
            return "I don't have enough information."
        context = "\n---\n".join(context_list)
        completion = (
            oai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a national parks expert.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                    },
                ],
            )
            .choices[0]
            .message.content
        )
        return completion or "No answer."

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        ctx = self.retrieve(query=query)
        return self.generate_completion(query=query, context_list=ctx)


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

provider = fOpenAI(model_engine="gpt-4o-mini")

f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons_consider_answerability,
    name="Groundedness",
    selectors={
        "source": Selector.select_context(collect_list=True),
        "statement": Selector.select_record_output(),
        "question": Selector.select_record_input(),
    },
)
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
    agg=np.mean,
)

METRICS = [f_groundedness, f_answer_relevance, f_context_relevance]


def make_run(label, run_suffix, invocation_workers=None, metric_workers=None):
    snowflake_connector = SnowflakeConnector(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        database=os.environ.get("SNOWFLAKE_DATABASE", "TRULENS_BENCHMARK"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    )
    session = TruSession(connector=snowflake_connector)

    rag = RAG(model_name="gpt-4o-mini")
    tru_rag = TruApp(
        rag,
        app_name="SF Bench App",
        app_version=label,
        main_method=rag.query,
        feedbacks=METRICS,
    )

    run = tru_rag.add_run(
        run_config=RunConfig(
            run_name=f"bench_{run_suffix}",
            dataset_name="test_questions",
            source_type="DATAFRAME",
            dataset_spec={"input": "input"},
            invocation_max_workers=invocation_workers,
            metric_max_workers=metric_workers,
        )
    )
    return run, session


def bench_start(run):
    t0 = time.perf_counter()
    run.start(input_df=test_dataset)
    return time.perf_counter() - t0


def bench_metrics(run):
    t0 = time.perf_counter()
    run.compute_metrics(METRICS)
    return time.perf_counter() - t0


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK: Parallel vs Sequential (Snowflake)")
    print(f"Dataset: {len(test_dataset)} rows, Metrics: {len(METRICS)}")
    print("=" * 60)

    print("\n--- run.start() benchmark ---")

    run_seq, _ = make_run("seq-invoke", "seq", invocation_workers=1)
    t_seq = bench_start(run_seq)
    print(f"  Sequential (workers=1): {t_seq:.2f}s")

    run_par, _ = make_run("par-invoke", "par", invocation_workers=None)
    t_par = bench_start(run_par)
    print(f"  Parallel   (default) : {t_par:.2f}s")

    speedup_start = t_seq / t_par if t_par > 0 else float("inf")
    print(f"  Speedup: {speedup_start:.2f}x")

    print("\n--- run.compute_metrics() benchmark ---")

    run_mseq, _ = make_run("seq-metric", "mseq", metric_workers=1)
    bench_start(run_mseq)
    t_met_seq = bench_metrics(run_mseq)
    print(f"  Sequential (workers=1): {t_met_seq:.2f}s")

    run_mpar, _ = make_run("par-metric", "mpar", metric_workers=None)
    bench_start(run_mpar)
    t_met_par = bench_metrics(run_mpar)
    print(f"  Parallel   (default) : {t_met_par:.2f}s")

    speedup_metrics = t_met_seq / t_met_par if t_met_par > 0 else float("inf")
    print(f"  Speedup: {speedup_metrics:.2f}x")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(
        f"  run.start()          : {t_seq:.2f}s -> {t_par:.2f}s  ({speedup_start:.2f}x)"
    )
    print(
        f"  run.compute_metrics(): {t_met_seq:.2f}s -> {t_met_par:.2f}s  ({speedup_metrics:.2f}x)"
    )
    print("=" * 60)
