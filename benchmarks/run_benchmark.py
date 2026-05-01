"""
Benchmark: Parallel vs Sequential Run APIs with Snowflake Cortex + Snowflake connector.

Benchmarks three things:
  1. run.start() parallelization (invocation_max_workers)
  2. run.compute_metrics() with client-side trace-level metrics (execution_efficiency, logical_consistency)
  3. run.compute_metrics() with server-side string metrics (groundedness, context_relevance)

Waits for OTEL spans to be ingested into the Snowflake event table before computing metrics.

Usage:
  SNOWFLAKE_CONNECTION_NAME=DEVREL_ENTERPRISE python -u benchmarks/run_benchmark.py
"""

import os
import time
import uuid

import chromadb
from openai import OpenAI
import pandas as pd
from snowflake.snowpark import Session
import toml
from trulens.apps.app import TruApp
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import Metric
from trulens.core import Selector
from trulens.core import TruSession
from trulens.core.otel.instrument import instrument
from trulens.core.run import RunConfig
from trulens.core.run import RunStatus
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
vector_store = chroma_client.get_or_create_collection(
    name="national_parks_bench_sf"
)
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

CLIENT_METRICS = [
    Metric(
        implementation=provider.execution_efficiency_with_cot_reasons,
        name="Execution Efficiency",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.logical_consistency_with_cot_reasons,
        name="Logical Consistency",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.tool_selection_with_cot_reasons,
        name="Tool Selection",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.plan_quality_with_cot_reasons,
        name="Plan Quality",
        selectors={"trace": Selector(trace_level=True)},
    ),
]

SERVER_METRICS = ["groundedness", "context_relevance"]


def get_snowflake_connector():
    snowpark_session = (
        Session.builder.config("connection_name", CONN_NAME)
        .config("database", "TRULENS_TEST")
        .config("schema", "PUBLIC")
        .config("warehouse", "COMPUTE")
        .create()
    )
    return SnowflakeConnector(snowpark_session=snowpark_session)


def wait_for_ingestion(run, timeout=120, poll_interval=5):
    ready_statuses = {
        RunStatus.INVOCATION_COMPLETED,
        RunStatus.INVOCATION_PARTIALLY_COMPLETED,
        RunStatus.COMPUTATION_IN_PROGRESS,
        RunStatus.COMPLETED,
        RunStatus.PARTIALLY_COMPLETED,
    }
    start = time.time()
    while time.time() - start < timeout:
        status = run.get_status()
        print(f"    Waiting for ingestion... status={status}")
        if status in ready_statuses:
            return status
        time.sleep(poll_interval)
    print(
        f"    WARNING: Timed out after {timeout}s, proceeding anyway (status={status})"
    )
    return status


def make_run(
    connector,
    session,
    label,
    metrics_list,
    invocation_workers=None,
    metric_workers=None,
):
    uid = uuid.uuid4().hex[:6]

    rag = RAG()
    tru_rag = TruApp(
        rag,
        connector=connector,
        app_name="Parallel Benchmark SF",
        app_version=label,
        main_method=rag.query,
        feedbacks=CLIENT_METRICS,
        start_evaluator=False,
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
    return run, metrics_list


def bench_start(run):
    t0 = time.perf_counter()
    run.start(input_df=test_dataset)
    return time.perf_counter() - t0


def bench_metrics(run, metrics):
    t0 = time.perf_counter()
    run.compute_metrics(metrics)
    return time.perf_counter() - t0


if __name__ == "__main__":
    connector = get_snowflake_connector()
    session = TruSession(connector=connector)

    print("=" * 70)
    print(f"BENCHMARK: Snowflake Cortex {CORTEX_MODEL} + Snowflake connector")
    print(f"Dataset: {len(test_dataset)} rows")
    print(f"Client-side metrics: {[m.name for m in CLIENT_METRICS]}")
    print(f"Server-side metrics: {SERVER_METRICS}")
    print("=" * 70)

    # ================================================================
    # 1. run.start() benchmark
    # ================================================================
    print("\n--- run.start() benchmark ---")

    run_seq, _ = make_run(
        connector, session, "seq_invoke", CLIENT_METRICS, invocation_workers=1
    )
    t_seq = bench_start(run_seq)
    print(f"  Sequential (workers=1): {t_seq:.2f}s")

    run_par, _ = make_run(
        connector,
        session,
        "par_invoke",
        CLIENT_METRICS,
        invocation_workers=None,
    )
    t_par = bench_start(run_par)
    print(f"  Parallel   (default) : {t_par:.2f}s")

    speedup_start = t_seq / t_par if t_par > 0 else float("inf")
    print(f"  Speedup: {speedup_start:.2f}x")

    # ================================================================
    # 2. Client-side metrics benchmark (execution_efficiency, logical_consistency)
    # ================================================================
    print("\n--- run.compute_metrics() — client-side (trace-level) ---")
    print(f"  Metrics: {[m.name for m in CLIENT_METRICS]}")
    print(
        "  (waiting for event table ingestion before each compute_metrics call)"
    )

    run_cm_seq, _ = make_run(
        connector, session, "seq_client_met", CLIENT_METRICS, metric_workers=1
    )
    bench_start(run_cm_seq)
    wait_for_ingestion(run_cm_seq)
    t_cm_seq = bench_metrics(run_cm_seq, CLIENT_METRICS)
    print(f"  Sequential (workers=1): {t_cm_seq:.2f}s")

    run_cm_par, _ = make_run(
        connector,
        session,
        "par_client_met",
        CLIENT_METRICS,
        metric_workers=None,
    )
    bench_start(run_cm_par)
    wait_for_ingestion(run_cm_par)
    t_cm_par = bench_metrics(run_cm_par, CLIENT_METRICS)
    print(f"  Parallel   (default) : {t_cm_par:.2f}s")

    speedup_client = t_cm_seq / t_cm_par if t_cm_par > 0 else float("inf")
    print(f"  Speedup: {speedup_client:.2f}x")

    # ================================================================
    # 3. Server-side metrics (parallelized by Snowflake, poll until done)
    # ================================================================
    print("\n--- run.compute_metrics() — server-side (Snowflake-managed) ---")
    print(f"  Metrics: {SERVER_METRICS}")
    print(
        "  (waiting for event table ingestion, then polling until results available)"
    )

    run_srv, _ = make_run(connector, session, "server_met", SERVER_METRICS)
    bench_start(run_srv)
    wait_for_ingestion(run_srv)

    t0_srv = time.perf_counter()
    run_srv.compute_metrics(SERVER_METRICS)
    done_statuses = {RunStatus.COMPLETED, RunStatus.PARTIALLY_COMPLETED}
    poll_timeout = 600
    poll_start = time.time()
    while time.time() - poll_start < poll_timeout:
        srv_status = run_srv.get_status()
        print(f"    Polling server-side metrics... status={srv_status}")
        if srv_status in done_statuses:
            break
        time.sleep(10)
    t_srv = time.perf_counter() - t0_srv
    print(f"  Server-side (dispatch + completion): {t_srv:.2f}s")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(
        f"  run.start()                    : {t_seq:.2f}s -> {t_par:.2f}s  ({speedup_start:.2f}x)"
    )
    print(
        f"  compute_metrics() client-side  : {t_cm_seq:.2f}s -> {t_cm_par:.2f}s  ({speedup_client:.2f}x)"
    )
    print(
        f"  compute_metrics() server-side  : {t_srv:.2f}s (Snowflake-managed, end-to-end)"
    )
    print("=" * 70)
