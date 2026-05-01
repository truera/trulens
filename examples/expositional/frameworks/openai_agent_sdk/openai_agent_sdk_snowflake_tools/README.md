# OpenAI Agent SDK + Snowflake Tools with TruLens Observability

A support intelligence agent built with the **OpenAI Agent SDK**, using **Snowflake Cortex Analyst** (structured data) and **Cortex Search** (unstructured knowledge base) as tools — fully instrumented with **TruLens** for tracing and evaluation via **Snowflake AI Observability**.

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────┐
│  React Chat  │────▶│  FastAPI (server.py)                         │
│  Frontend    │◀────│  @trace_with_run                             │
└──────────────┘     └──────────────────────┬───────────────────────┘
                                            │
┌──────────────┐                            │
│  Batch Eval  │                            │
│  run.start() │────────────────────────────┤
└──────────────┘                            │
                                            │
                     ┌──────────────────────▼───────────────────────┐   ┌─────────────────────────┐
                     │  OpenAI Agent SDK — AgentApp.ask(question)   │   │                         │
                     │  ┌────────────────────────────────────────┐  │   │  Snowflake Event Table  │
                     │  │  TruLens @instrument (OTel spans)      │──┼──▶│  (OTel trace logs)      │
                     │  │  RECORD_ROOT · TOOL · RETRIEVAL · GEN  │  │   │                         │
                     │  └────────────────────────────────────────┘  │   └─────────────────────────┘
                     └─────────────┬───────────────────┬────────────┘
                                   │                   │
                      function_call│                   │function_call
                                   │                   │
              ┌────────────────────▼──┐   ┌────────────▼────────────┐
              │  query_ticket_        │   │  search_knowledge_      │
              │  metrics              │   │  base                   │
              │  (Cortex Analyst)     │   │  (Cortex Search)        │
              └──────────┬───────────┘   └────────────┬─────────────┘
                         │                            │
              ┌──────────▼───────────┐   ┌────────────▼─────────────┐
              │  TICKET_METRICS      │   │  KB_ARTICLES             │
              └──────────────────────┘   └──────────────────────────┘
```

## Two Invocation Modes

### 1. Batch Evaluation (`run_eval.py`)

Runs all 23 test queries (13 analyst + 5 search + 5 mixed) through the agent, then computes metrics:

```bash
SNOWFLAKE_CONNECTION_NAME=YOUR_CONNECTION python run_eval.py
```

```python
run_config = RunConfig(
    run_name=f"BATCH_EVAL_RUN_{timestamp}",
    dataset_spec={"input": "question"},
    llm_judge_name="openai-gpt-5.1",
)
run = tru_app.add_run(run_config=run_config)
run.start(input_df=input_df)
run.compute_metrics(metrics=all_metrics)
```

### 2. Production Monitoring (`server.py`)

A FastAPI + React chat UI for interactive use. Each message is traced in real time via `@trace_with_run`:

```bash
SNOWFLAKE_CONNECTION_NAME=YOUR_CONNECTION uvicorn server:app --reload --port 8000
cd frontend && npm run dev
```

```python
@trace_with_run(app=tru_app, run_name=run_name)
def process_message(message: str):
    return agent_app.ask(message)
```

## Observability

### Tracing with `@instrument`

Every call to `agent_app.ask()` produces a trace with typed spans. The `@instrument` decorator from TruLens wraps each function and emits OpenTelemetry spans with semantic attributes that flow to Snowflake's event table:

```
RECORD_ROOT span (AgentApp.ask)
│   input: question  |  output: final_answer
│
├── GENERATION span (SnowflakeAsyncOpenAI.create)
│       input_messages | model | tool_calls | token usage
│
├── TOOL span (ask_database)
│       kwargs: question  |  return: SQL + results
│
├── RETRIEVAL span (search_knowledge_base)
│       query_text: query  |  retrieved_contexts: [chunks]
│
└── GENERATION span (SnowflakeAsyncOpenAI.create)
        input_messages (with tool results) | content: final answer
```

Usage in tool functions:

```python
@function_tool
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    },
)
def search_knowledge_base(query: str) -> str:
    ...
```

The span attributes (`RETRIEVAL.QUERY_TEXT`, `RETRIEVAL.RETRIEVED_CONTEXTS`, `RECORD_ROOT.INPUT`, `RECORD_ROOT.OUTPUT`) are what evaluation metrics consume downstream.

### Evaluation: Server-Side Metrics

Snowflake AI Observability provides built-in metrics that run entirely server-side using Cortex LLMs as judges. Pass metric names as strings — no provider setup needed:

```python
run.compute_metrics(metrics=[
    "context_relevance",
    "groundedness",
    "answer_relevance",
    "coherence",
])
```

| Metric | What It Measures | Required Span Attributes |
|--------|-----------------|--------------------------|
| `"context_relevance"` | Is retrieved context relevant to the query? | `RETRIEVAL.QUERY_TEXT`, `RETRIEVAL.RETRIEVED_CONTEXTS` |
| `"groundedness"` | Is the response supported by retrieved context? | `RETRIEVAL.RETRIEVED_CONTEXTS`, `RECORD_ROOT.OUTPUT` |
| `"answer_relevance"` | Is the response relevant to the question? | `RECORD_ROOT.INPUT`, `RECORD_ROOT.OUTPUT` |
| `"coherence"` | Is the response logically coherent? | `RECORD_ROOT.OUTPUT` |

### Evaluation: Custom Client-Side Metrics

For domain-specific evaluations, custom metrics are defined in `src/eval/metrics.py` using the TruLens `Metric` class with `Selector` to extract data from specific span types. All metrics are wired up via the `build_metrics()` factory function:

```python
from trulens.providers.cortex import Cortex

provider = Cortex(snowpark_session=session, model_engine="openai-gpt-5.1")
```

| Metric | Type | What It Measures |
|--------|------|-----------------|
| SQL Agreement (Golden) | Result-set comparison | Does the generated SQL produce the same results as golden SQL? |
| Precision@k | Ground truth | Fraction of retrieved chunks that are expected |
| Recall@k | Ground truth | Fraction of expected chunks that were retrieved |

## Ground Truth

Ground truth is stored in Snowflake tables (not hardcoded in Python), making it editable in Snowsight, versionable via `CREATE TABLE ... CLONE`, and shareable across evaluation runs.

### Cortex Search (Retrieval)

Table `GROUND_TRUTH_SEARCH` stores expected retrieval chunks per query:

```sql
CREATE TABLE GROUND_TRUTH_SEARCH (
    QUERY VARCHAR,              -- The user question
    EXPECTED_CHUNK_TEXT TEXT,    -- Expected chunk (matches Cortex Search return format)
    EXPECT_SCORE FLOAT          -- Relevance weight (default 1.0)
);
```

Loaded at startup by `load_search_ground_truth()` in `src/ground_truth.py`, which groups rows by query into the `[{query, expected_chunks: [{text, expect_score}]}]` format that `GroundTruthAgreement` expects. This feeds **Precision@k** and **Recall@k** metrics.

### Cortex Analyst (Golden SQL)

Table `GROUND_TRUTH_ANALYST` stores golden SQL per query:

```sql
CREATE TABLE GROUND_TRUTH_ANALYST (
    QUERY VARCHAR,      -- The user question
    GOLDEN_SQL TEXT      -- The reference SQL that produces correct results
);
```

Golden SQL entries can come from any source — `verified_queries` in the semantic model, hand-written reference queries, or any SQL you know produces correct results. The **SQL Agreement (Golden)** metric works as follows:

1. Look up the golden SQL for the given question
2. Extract the predicted SQL from the agent's tool output
3. Execute both SQL queries against Snowflake
4. Compare the two result DataFrames using the structural comparator (`src/sql_result_comparator.py`)

### SQL Result Comparator

The comparator (`src/sql_result_comparator.py`) uses a multi-strategy approach to handle differences in column naming, row ordering, null handling, and floating-point precision:

```
1. Convert string columns to numeric where possible
2. Early return if DataFrames are exactly equal
3. Align columns using LLM-based semantic matching (Cortex Complete)
4. Try 4 normalization strategies:
   - No modifications
   - Remove null rows
   - Sort rows by non-categorical columns
   - Remove nulls + sort
5. For each strategy, compare columns using:
   - Exact match
   - Approximate match (epsilon tolerance for floats)
   - Categorical bijection (1:1 value mapping)
6. Return the best result across all strategies
```

Scoring: `Equal` = 1.0, `Need manual analysis` = 0.5, `Not equal` = 0.0.

### Maintaining Ground Truth

To add new ground truth entries:

```sql
-- Add a new search ground truth entry
INSERT INTO GROUND_TRUTH_SEARCH VALUES
('How do I manage team members?',
 '[account] Managing Team Members and Roles\nTo manage team members...',
 1.0);

-- Add a new analyst golden SQL entry
INSERT INTO GROUND_TRUTH_ANALYST VALUES
('What is the resolution rate for billing issues?',
 'SELECT COUNT(CASE WHEN STATUS IN (''resolved'', ''closed'') THEN 1 END) * 100.0 / COUNT(TICKET_ID) AS RESOLUTION_RATE FROM TICKET_METRICS WHERE CATEGORY = ''billing''');
```

Version ground truth for A/B evaluation:

```sql
CREATE TABLE GROUND_TRUTH_ANALYST_V2 CLONE GROUND_TRUTH_ANALYST;
```

## Prerequisites

1. **Snowflake account** with Cortex features enabled
2. **Snowflake PAT** or a named connection in `~/.snowflake/connections.toml`
3. **Python 3.12** and [uv](https://docs.astral.sh/uv/)
4. **Node.js 18+** (for the React frontend)
5. **Snowflake CLI v3.14.0+** (for dashboard deployment) — or use `uvx` to run the latest version automatically

## Setup

### 1. Configure Snowflake Connection

```bash
export SNOWFLAKE_CONNECTION_NAME="YOUR_CONNECTION_NAME"
```

### 2. Create Snowflake Objects

```bash
snowsql -f setup_snowflake.sql
```

```sql
PUT file://semantic_model.yaml @SUPPORT_INTELLIGENCE.DATA.MODELS AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

### 3. Install Dependencies

```bash
uv sync
cd frontend && npm install
```

### 4. Deploy Monitoring Dashboard

Deploy the Streamlit observability dashboard to Snowflake:

```bash
cd monitoring_dashboard
uvx --from snowflake-cli snow streamlit deploy --replace --connection $SNOWFLAKE_CONNECTION_NAME
```

> **Note:** `uvx` runs the latest Snowflake CLI in an isolated environment, avoiding version or dependency conflicts. The SPCS container runtime (`runtime_name` + `compute_pool` in `snowflake.yml`) requires CLI v3.14.0+. If you have a compatible version installed locally, you can use `snow` directly instead of `uvx --from snowflake-cli snow`.

Once deployed, open the app in Snowsight at **SUPPORT_INTELLIGENCE.DATA.AI_OBSERVABILITY_DASHBOARD**.

## File Structure

```
server.py                            # FastAPI backend — production monitoring with @trace_with_run
run_eval.py                          # Batch evaluation runner with metric computation
setup_snowflake.sql                  # Snowflake DDL: tables, search service, stage
semantic_model.yaml                  # Cortex Analyst semantic model for TICKET_METRICS
pyproject.toml                       # All dependencies (uv-managed)
src/
    agent/
        app.py                       # OpenAI Agent SDK agent + SnowflakeAsyncOpenAI wrapper
        tools.py                     # @function_tool + @instrument tools (Analyst + Search)
    services/
        config.py                    # Snowflake connection params and session factory
    eval/
        ground_truth.py              # Ground truth loaders (from Snowflake tables) + test query suites
        metrics.py                   # All metric implementations + build_metrics() factory (SQL Agreement, Precision@k, Recall@k)
        sql_result_comparator.py     # SQL result-set comparator for golden SQL evaluation
    observability/
        trulens_setup.py             # TruLens setup: TruApp, SnowflakeConnector, metric wiring
frontend/
    src/App.tsx                      # React chat UI with run_name persistence
monitoring_dashboard/
    streamlit_app.py                 # Custom Streamlit observability dashboard (Snowsight)
    snowflake.yml                    # Streamlit-in-Snowflake deployment config
    pyproject.toml                   # Dashboard dependencies
```

## Sample Questions

**Ticket Metrics (Cortex Analyst):** "How many tickets are there by priority?" · "Which agent has the highest CSAT score?" · "What is the average resolution time for technical tickets?"

**Knowledge Base (Cortex Search):** "How do I reset my password?" · "What are the API rate limits?" · "How do I configure webhooks?"

**Mixed (both tools):** "How many high priority tickets do we have and what's our SLA?" · "What is the average CSAT score and how do customers reset passwords?"
