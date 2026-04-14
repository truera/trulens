"""
Benchmark: Parallel vs Sequential Run APIs with Snowflake connection.

Uses the batch_evaluation quickstart RAG app (ChromaDB + OpenAI).
Connects to Snowflake via the DEVREL_ENTERPRISE connection.

Usage:
  OPENAI_API_KEY=... SNOWFLAKE_CONNECTION_NAME=DEVREL_ENTERPRISE \
    python benchmarks/run_benchmark.py
"""

import os
import time
import uuid

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import numpy as np
from openai import OpenAI
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
from trulens.providers.openai import OpenAI as fOpenAI

# --- Knowledge Base ---

yellowstone_info = """
Yellowstone National Park, established in 1872, was the world's first national park.
Located primarily in Wyoming, it spans nearly 3,500 square miles and sits atop a volcanic hotspot.
The park is famous for its geothermal features, including Old Faithful geyser, and is home to
grizzly bears, wolves, bison, and elk.
"""
yosemite_info = """
Yosemite National Park is located in California's Sierra Nevada mountains.
It is known for its granite cliffs, waterfalls, giant sequoia groves, and biological diversity.
El Capitan and Half Dome are among the most iconic rock formations in the world.
The park covers about 750,000 acres and was instrumental in the development of the national park idea.
"""
grand_canyon_info = """
The Grand Canyon, carved by the Colorado River over millions of years, is one of the most
spectacular geological features on Earth. Located in Arizona, the canyon is 277 miles long,
up to 18 miles wide, and over a mile deep. Grand Canyon National Park was established in 1919.
"""
zion_info = """
Zion National Park is located in southwestern Utah and is known for its steep red cliffs.
The park's main feature is Zion Canyon, which is 15 miles long and up to 2,640 feet deep.
The Virgin River runs through the canyon. Popular hikes include Angels Landing and The Narrows.
"""
glacier_info = """
Glacier National Park in Montana contains over 700 miles of hiking trails, numerous glacially
carved lakes, and the famous Going-to-the-Sun Road. The park is part of the Crown of the
Continent ecosystem and is home to grizzly bears, mountain goats, and wolverines.
"""
acadia_info = """
Acadia National Park, located on Mount Desert Island in Maine, protects the natural beauty
of the highest rocky headlands along the Atlantic coastline. The park includes woodlands,
rocky beaches, and glacier-scoured granite peaks such as Cadillac Mountain.
"""

# --- ChromaDB ---

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection(
    name="national_parks_bench", embedding_function=embedding_function
)
vector_store.add("yellowstone", documents=yellowstone_info)
vector_store.add("yosemite", documents=yosemite_info)
vector_store.add("grand_canyon", documents=grand_canyon_info)
vector_store.add("zion", documents=zion_info)
vector_store.add("glacier", documents=glacier_info)
vector_store.add("acadia", documents=acadia_info)

oai_client = OpenAI()


# --- RAG App ---


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
            return "I don't have enough information to answer this question."
        context = "\n---\n".join(context_list)
        completion = (
            oai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a national parks expert. Answer questions using only the provided context.",
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
        return completion or "No answer generated."

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        context_list = self.retrieve(query=query)
        return self.generate_completion(query=query, context_list=context_list)


# --- Test Dataset ---

test_dataset = pd.DataFrame({
    "input": [
        "When was Yellowstone established and where is it located?",
        "What are the famous rock formations in Yosemite?",
        "How deep is the Grand Canyon?",
        "What river runs through Zion Canyon?",
        "What wildlife can be found in Glacier National Park?",
        "Where is Acadia National Park located?",
        "Which national park has geothermal features?",
        "What is the Going-to-the-Sun Road?",
    ]
})

# --- Metrics ---

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


def get_snowflake_connector():
    conn_name = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "DEVREL_ENTERPRISE")
    snowpark_session = (
        Session.builder.config("connection_name", conn_name)
        .config("database", "TRULENS_TEST")
        .config("schema", "PUBLIC")
        .config("warehouse", "COMPUTE")
        .create()
    )
    return SnowflakeConnector(
        snowpark_session=snowpark_session,
    )


def make_run(label, invocation_workers=None, metric_workers=None):
    uid = uuid.uuid4().hex[:6]
    connector = get_snowflake_connector()
    session = TruSession(connector=connector)

    rag = RAG(model_name="gpt-4o-mini")
    tru_rag = TruApp(
        rag,
        app_name="Parallel Benchmark",
        app_version=label,
        main_method=rag.query,
        feedbacks=METRICS,
    )

    run = tru_rag.add_run(
        run_config=RunConfig(
            run_name=f"bench_{label}_{uid}",
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

    # --- run.start() benchmark ---
    print("\n--- run.start() benchmark ---")

    run_seq, _ = make_run("seq_invoke", invocation_workers=1)
    t_seq = bench_start(run_seq)
    print(f"  Sequential (workers=1): {t_seq:.2f}s")

    run_par, _ = make_run("par_invoke", invocation_workers=None)
    t_par = bench_start(run_par)
    print(f"  Parallel   (default) : {t_par:.2f}s")

    speedup_start = t_seq / t_par if t_par > 0 else float("inf")
    print(f"  Speedup: {speedup_start:.2f}x")

    # --- run.compute_metrics() benchmark ---
    print("\n--- run.compute_metrics() benchmark ---")

    run_mseq, _ = make_run("seq_metric", metric_workers=1)
    bench_start(run_mseq)
    t_met_seq = bench_metrics(run_mseq)
    print(f"  Sequential (workers=1): {t_met_seq:.2f}s")

    run_mpar, _ = make_run("par_metric", metric_workers=None)
    bench_start(run_mpar)
    t_met_par = bench_metrics(run_mpar)
    print(f"  Parallel   (default) : {t_met_par:.2f}s")

    speedup_metrics = t_met_seq / t_met_par if t_met_par > 0 else float("inf")
    print(f"  Speedup: {speedup_metrics:.2f}x")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(
        f"  run.start()          : {t_seq:.2f}s -> {t_par:.2f}s  ({speedup_start:.2f}x)"
    )
    print(
        f"  run.compute_metrics(): {t_met_seq:.2f}s -> {t_met_par:.2f}s  ({speedup_metrics:.2f}x)"
    )
    print("=" * 60)
