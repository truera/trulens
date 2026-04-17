"""
Benchmark: Parallel vs Sequential Run APIs — TruLens OSS only (no Snowflake).

Uses Cortex LLM via OpenAI compat for both RAG app and metric evaluation.
No Snowflake connector — uses default TruSession (SQLite) so events are in-memory.

Usage:
  SNOWFLAKE_CONNECTION_NAME=DEVREL_ENTERPRISE python benchmarks/run_benchmark_oss.py
"""

import os
import time
import uuid

import chromadb
import numpy as np
from openai import OpenAI
import pandas as pd
import toml
from trulens.apps.app import TruApp
from trulens.core import Metric
from trulens.core import Selector
from trulens.core import TruSession
from trulens.core.otel.instrument import instrument
from trulens.core.run import RunConfig
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import OpenAI as fOpenAI

os.environ.pop("OPENAI_API_KEY", None)

CONN_NAME = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "DEVREL_ENTERPRISE")
CORTEX_MODEL = "claude-sonnet-4-6"

connections_path = os.path.expanduser("~/.snowflake/connections.toml")
conn_cfg = toml.load(connections_path)[CONN_NAME]
SF_ACCOUNT = conn_cfg["account"]
SF_PAT = conn_cfg["password"]
CORTEX_BASE_URL = (
    f"https://{SF_ACCOUNT}.snowflakecomputing.com/api/v2/cortex/v1"
)

cortex_client = OpenAI(api_key=SF_PAT, base_url=CORTEX_BASE_URL)

KNOWLEDGE = {
    "yellowstone": """
Yellowstone National Park, established in 1872, was the world's first national park.
Located primarily in Wyoming, it spans nearly 3,500 square miles and sits atop a volcanic hotspot.
The park is famous for its geothermal features, including Old Faithful geyser, and is home to
grizzly bears, wolves, bison, and elk.
""",
    "yosemite": """
Yosemite National Park is located in California's Sierra Nevada mountains.
It is known for its granite cliffs, waterfalls, giant sequoia groves, and biological diversity.
El Capitan and Half Dome are among the most iconic rock formations in the world.
The park covers about 750,000 acres and was instrumental in the development of the national park idea.
""",
    "grand_canyon": """
The Grand Canyon, carved by the Colorado River over millions of years, is one of the most
spectacular geological features on Earth. Located in Arizona, the canyon is 277 miles long,
up to 18 miles wide, and over a mile deep. Grand Canyon National Park was established in 1919.
""",
    "zion": """
Zion National Park is located in southwestern Utah and is known for its steep red cliffs.
The park's main feature is Zion Canyon, which is 15 miles long and up to 2,640 feet deep.
The Virgin River runs through the canyon. Popular hikes include Angels Landing and The Narrows.
""",
    "glacier": """
Glacier National Park in Montana contains over 700 miles of hiking trails, numerous glacially
carved lakes, and the famous Going-to-the-Sun Road. The park is part of the Crown of the
Continent ecosystem and is home to grizzly bears, mountain goats, and wolverines.
""",
    "acadia": """
Acadia National Park, located on Mount Desert Island in Maine, protects the natural beauty
of the highest rocky headlands along the Atlantic coastline. The park includes woodlands,
rocky beaches, and glacier-scoured granite peaks such as Cadillac Mountain.
""",
}

chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection(name="national_parks_oss")
for doc_id, doc_text in KNOWLEDGE.items():
    vector_store.add(doc_id, documents=doc_text)


class RAG:
    def __init__(self, model_name=CORTEX_MODEL):
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
            cortex_client.chat.completions.create(
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

cortex_eval_client = OpenAI(api_key=SF_PAT, base_url=CORTEX_BASE_URL)
provider = fOpenAI(
    model_engine=CORTEX_MODEL,
    client=cortex_eval_client,
)
provider._set_capabilities({
    "structured_outputs": False,
    "cfg": False,
})

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
f_coherence = Metric(
    implementation=provider.coherence_with_cot_reasons,
    name="Coherence",
    selectors={
        "text": Selector.select_record_output(),
    },
)

METRICS = [f_groundedness, f_answer_relevance, f_context_relevance, f_coherence]


def make_run(session, label, invocation_workers=None, metric_workers=None):
    uid = uuid.uuid4().hex[:6]

    rag = RAG()
    tru_rag = TruApp(
        rag,
        connector=session.connector,
        app_name="Parallel Benchmark OSS",
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
    return run


def bench_start(run):
    t0 = time.perf_counter()
    run.start(input_df=test_dataset)
    return time.perf_counter() - t0


def bench_metrics(run):
    t0 = time.perf_counter()
    run.compute_metrics(METRICS)
    return time.perf_counter() - t0


if __name__ == "__main__":
    session = TruSession()
    session.reset_database()

    print("=" * 60)
    print(
        f"BENCHMARK: Parallel vs Sequential — OSS (Cortex {CORTEX_MODEL}, no SF connector)"
    )
    print(f"Dataset: {len(test_dataset)} rows, Metrics: {len(METRICS)}")
    print("=" * 60)

    # --- run.start() benchmark ---
    print("\n--- run.start() benchmark ---")

    run_seq = make_run(session, "seq_invoke", invocation_workers=1)
    t_seq = bench_start(run_seq)
    print(f"  Sequential (workers=1): {t_seq:.2f}s")

    run_par = make_run(session, "par_invoke", invocation_workers=None)
    t_par = bench_start(run_par)
    print(f"  Parallel   (default) : {t_par:.2f}s")

    speedup_start = t_seq / t_par if t_par > 0 else float("inf")
    print(f"  Speedup: {speedup_start:.2f}x")

    # --- run.compute_metrics() benchmark ---
    print("\n--- run.compute_metrics() benchmark ---")

    run_mseq = make_run(session, "seq_metric", metric_workers=1)
    bench_start(run_mseq)
    t_met_seq = bench_metrics(run_mseq)
    print(f"  Sequential (workers=1): {t_met_seq:.2f}s")

    run_mpar = make_run(session, "par_metric", metric_workers=None)
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
