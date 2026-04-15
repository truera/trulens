import json
import threading

import requests
from agents import function_tool
from snowflake.core import Root
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from src.services.config import (
    CORTEX_SEARCH_SERVICE,
    SEMANTIC_MODEL_FILE,
    SNOWFLAKE_ACCOUNT_URL,
    SNOWFLAKE_PAT,
    get_snowpark_session,
)

_session = get_snowpark_session()
_search_root = Root(_session)

_analyst_tls = threading.local()


def _analyst_attributes(ret, exception, *args, **kwargs):
    structured = getattr(_analyst_tls, "last_result", None)
    attrs = {
        SpanAttributes.CALL.KWARGS: str(kwargs.get("question", args[0] if args else "")),
        SpanAttributes.CALL.RETURN: str(ret),
    }
    if structured:
        attrs["ai.observability.analyst.interpretation"] = structured.get("interpretation", "")
        attrs["ai.observability.analyst.generated_sql"] = structured.get("generated_sql", "")
        attrs["ai.observability.analyst.query_results"] = structured.get("query_results", "")
    return attrs


@function_tool
@instrument(
    name="ask_database",
    span_type=SpanAttributes.SpanType.TOOL,
    attributes=_analyst_attributes,
)
def ask_database(question: str) -> str:
    """Query structured data using natural language via Cortex Analyst.
    Use this tool when users ask quantitative questions that can be
    answered with SQL against the semantic model."""
    resp = requests.post(
        f"{SNOWFLAKE_ACCOUNT_URL}/api/v2/cortex/analyst/message",
        headers={
            "Authorization": f"Bearer {SNOWFLAKE_PAT}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        },
        json={
            "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
            "semantic_model_file": SEMANTIC_MODEL_FILE,
        },
    )
    resp.raise_for_status()
    data = resp.json()

    message = data.get("message", {})
    content_blocks = message.get("content", [])

    interpretation = ""
    generated_sql = ""
    query_results = ""

    for block in content_blocks:
        if block.get("type") == "text":
            interpretation = block["text"]
        elif block.get("type") == "sql":
            generated_sql = block["statement"]

    if generated_sql:
        try:
            result_df = _session.sql(generated_sql).to_pandas()
            query_results = result_df.to_string(index=False)
        except Exception as e:
            query_results = f"SQL execution error: {e}"

    _analyst_tls.last_result = {
        "interpretation": interpretation,
        "generated_sql": generated_sql,
        "query_results": query_results,
    }

    parts = []
    if interpretation:
        parts.append(interpretation)
    if generated_sql:
        parts.append(f"SQL: {generated_sql}")
    if query_results:
        parts.append(f"\nResults:\n{query_results}")

    return "\n".join(parts)


@function_tool
@instrument(
    name="search_knowledge_base",
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs.get("query", args[0] if args else ""),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: ret.split("\n\n---\n\n") if isinstance(ret, str) else [],
    },
)
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant articles and documentation.
    Use this tool when users ask how-to questions, need troubleshooting
    guidance, or want information from unstructured content."""
    db, schema, service = CORTEX_SEARCH_SERVICE.split(".")
    search_service = (
        _search_root.databases[db]
        .schemas[schema]
        .cortex_search_services[service]
    )
    results = search_service.search(
        query=query,
        columns=["TITLE", "CONTENT", "CATEGORY"],
        limit=1,
    )
    result_data = json.loads(results.to_json())
    items = result_data.get("results", [])
    if not items:
        return "No relevant knowledge base articles found."

    chunks = []
    for item in items:
        title = item.get("TITLE", "Untitled")
        content = item.get("CONTENT", "")
        category = item.get("CATEGORY", "")
        chunks.append(f"[{category}] {title}\n{content}")

    return "\n\n---\n\n".join(chunks)
