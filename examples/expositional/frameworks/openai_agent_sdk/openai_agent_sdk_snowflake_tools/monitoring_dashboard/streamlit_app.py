import json

import streamlit as st
import pandas as pd
import altair as alt
from datetime import timedelta

st.set_page_config(
    page_title="Agent Production Monitoring",
    page_icon=":bar_chart:",
    layout="wide",
)

conn = st.connection("snowflake")


@st.cache_data(ttl=timedelta(minutes=2))
def load_agents():
    return conn.query("""
        SELECT
            RECORD_ATTRIBUTES:"snow.ai.observability.object.name"::STRING AS agent_name,
            COUNT(*) AS total_spans
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
        WHERE RECORD_TYPE = 'SPAN'
          AND TIMESTAMP > DATEADD('day', -1, CURRENT_TIMESTAMP())
          AND RECORD_ATTRIBUTES:"snow.ai.observability.object.name"::STRING IS NOT NULL
        GROUP BY 1
        ORDER BY total_spans DESC
    """)


@st.cache_data(ttl=timedelta(minutes=2))
def load_runs(agent_name: str):
    return conn.query(f"""
        SELECT
            RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING AS run_name,
            MIN(TIMESTAMP) AS first_event,
            MAX(TIMESTAMP) AS last_event,
            COUNT(*) AS total_spans
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
        WHERE RECORD_TYPE = 'SPAN'
          AND TIMESTAMP > DATEADD('day', -1, CURRENT_TIMESTAMP())
          AND RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING IS NOT NULL
          AND RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING != ''
          AND RECORD_ATTRIBUTES:"snow.ai.observability.object.name"::STRING = '{agent_name}'
        GROUP BY 1
        ORDER BY first_event DESC
    """)


@st.cache_data(ttl=timedelta(minutes=2))
def load_spans(run_names: tuple):
    run_list = ",".join(f"'{r}'" for r in run_names)
    return conn.query(f"""
        SELECT
            TIMESTAMP,
            START_TIMESTAMP,
            RECORD_ATTRIBUTES:"ai.observability.span_type"::STRING AS span_type,
            RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING AS run_name,
            RECORD_ATTRIBUTES:"ai.observability.call.function"::STRING AS tool_function,
            RECORD_ATTRIBUTES:"ai.observability.record_root.input"::STRING AS input_question,
            RECORD_ATTRIBUTES:"ai.observability.record_root.output"::STRING AS output_answer,
            RECORD_ATTRIBUTES:"ai.observability.cost.model"::STRING AS model_name,
            RECORD_ATTRIBUTES:"ai.observability.cost.num_tokens"::INT AS total_tokens,
            RECORD_ATTRIBUTES:"ai.observability.cost.num_prompt_tokens"::INT AS prompt_tokens,
            RECORD_ATTRIBUTES:"ai.observability.cost.num_completion_tokens"::INT AS completion_tokens,
            TIMESTAMPDIFF('MILLISECOND', START_TIMESTAMP, TIMESTAMP) AS latency_ms,
            TRACE:"trace_id"::STRING AS trace_id,
            RECORD_ATTRIBUTES:"ai.observability.record_id"::STRING AS record_id
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
        WHERE RECORD_TYPE = 'SPAN'
          AND TIMESTAMP > DATEADD('day', -1, CURRENT_TIMESTAMP())
          AND RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING IN ({run_list})
        ORDER BY TIMESTAMP ASC
    """)


@st.cache_data(ttl=timedelta(minutes=2))
def load_eval_roots(run_names: tuple):
    run_list = ",".join(f"'{r}'" for r in run_names)
    return conn.query(f"""
        WITH run_record_ids AS (
            SELECT DISTINCT RECORD_ATTRIBUTES:"ai.observability.record_id"::STRING AS record_id
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
            WHERE RECORD_TYPE = 'SPAN'
              AND TIMESTAMP > DATEADD('day', -1, CURRENT_TIMESTAMP())
              AND RECORD_ATTRIBUTES:"ai.observability.span_type"::STRING = 'record_root'
              AND RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING IN ({run_list})
        )
        SELECT
            e.TIMESTAMP,
            e.RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING AS run_name,
            e.RECORD_ATTRIBUTES:"ai.observability.eval_root.metric_name"::STRING AS metric_name,
            e.RECORD_ATTRIBUTES:"ai.observability.eval_root.score"::FLOAT AS score,
            e.RECORD_ATTRIBUTES:"ai.observability.eval_root.higher_is_better"::BOOLEAN AS higher_is_better,
            e.RECORD_ATTRIBUTES:"ai.observability.eval.target_record_id"::STRING AS target_record_id
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS e
        WHERE e.RECORD_TYPE = 'SPAN'
          AND e.TIMESTAMP > DATEADD('day', -1, CURRENT_TIMESTAMP())
          AND e.RECORD_ATTRIBUTES:"ai.observability.span_type"::STRING = 'eval_root'
          AND (
              e.RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING IN ({run_list})
              OR e.RECORD_ATTRIBUTES:"ai.observability.eval.target_record_id"::STRING IN (SELECT record_id FROM run_record_ids)
          )
        ORDER BY e.TIMESTAMP ASC
    """)


@st.cache_data(ttl=timedelta(minutes=2))
def load_tool_details(run_names: tuple):
    run_list = ",".join(f"'{r}'" for r in run_names)
    return conn.query(f"""
        SELECT
            TIMESTAMP,
            RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING AS run_name,
            RECORD_ATTRIBUTES:"ai.observability.span_type"::STRING AS span_type,
            RECORD_ATTRIBUTES:"ai.observability.call.function"::STRING AS tool_function,
            RECORD_ATTRIBUTES:"ai.observability.call.kwargs.question"::STRING AS analyst_question,
            RECORD_ATTRIBUTES:"ai.observability.call.kwargs.query"::STRING AS search_query,
            RECORD_ATTRIBUTES:"ai.observability.call.return"::STRING AS call_return,
            RECORD_ATTRIBUTES:"ai.observability.analyst.interpretation"::STRING AS analyst_interpretation,
            RECORD_ATTRIBUTES:"ai.observability.analyst.generated_sql"::STRING AS analyst_generated_sql,
            RECORD_ATTRIBUTES:"ai.observability.analyst.query_results"::STRING AS analyst_query_results,
            RECORD_ATTRIBUTES:"ai.observability.retrieval.query_text"::STRING AS retrieval_query_text,
            RECORD_ATTRIBUTES:"ai.observability.retrieval.retrieved_contexts"::STRING AS retrieved_contexts_raw,
            TIMESTAMPDIFF('MILLISECOND', START_TIMESTAMP, TIMESTAMP) AS latency_ms,
            TRACE:"trace_id"::STRING AS trace_id
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
        WHERE RECORD_TYPE = 'SPAN'
          AND TIMESTAMP > DATEADD('day', -1, CURRENT_TIMESTAMP())
          AND RECORD_ATTRIBUTES:"ai.observability.span_type"::STRING IN ('tool', 'retrieval')
          AND RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING IN ({run_list})
        ORDER BY TIMESTAMP ASC
    """)


@st.cache_data(ttl=timedelta(minutes=2))
def load_server_evals(run_names: tuple):
    run_list = ",".join(f"'{r}'" for r in run_names)
    return conn.query(f"""
        WITH run_record_ids AS (
            SELECT DISTINCT RECORD_ATTRIBUTES:"ai.observability.record_id"::STRING AS record_id
            FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS
            WHERE RECORD_TYPE = 'SPAN'
              AND TIMESTAMP > DATEADD('day', -1, CURRENT_TIMESTAMP())
              AND RECORD_ATTRIBUTES:"ai.observability.span_type"::STRING = 'record_root'
              AND RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING IN ({run_list})
        )
        SELECT
            e.TIMESTAMP,
            e.RECORD:name::STRING AS span_name,
            e.RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING AS run_name,
            e.RECORD_ATTRIBUTES:"ai.observability.eval.metric_name"::STRING AS metric_name,
            e.RECORD_ATTRIBUTES:"ai.observability.eval.score"::FLOAT AS score,
            e.RECORD_ATTRIBUTES:"ai.observability.eval.criteria"::STRING AS criteria,
            e.RECORD_ATTRIBUTES:"ai.observability.eval.llm_judge_name"::STRING AS llm_judge,
            e.RECORD_ATTRIBUTES:"ai.observability.eval.target_record_id"::STRING AS target_record_id
        FROM SNOWFLAKE.LOCAL.AI_OBSERVABILITY_EVENTS e
        WHERE e.RECORD_TYPE = 'SPAN'
          AND e.TIMESTAMP > DATEADD('day', -1, CURRENT_TIMESTAMP())
          AND e.RECORD_ATTRIBUTES:"ai.observability.span_type"::STRING = 'eval'
          AND e.RECORD_ATTRIBUTES:"ai.observability.eval.metric_name"::STRING IS NOT NULL
          AND (
              e.RECORD_ATTRIBUTES:"ai.observability.run.name"::STRING IN ({run_list})
              OR e.RECORD_ATTRIBUTES:"ai.observability.eval.target_record_id"::STRING IN (SELECT record_id FROM run_record_ids)
          )
        ORDER BY e.TIMESTAMP ASC
    """)


st.title("Agent Production Monitoring")


def _get_selected_trace_id(event) -> str | None:
    if not event or not hasattr(event, "selection"):
        return None
    sel = event.selection
    if not sel:
        return None
    for key in sel:
        params = sel[key]
        if isinstance(params, list):
            for item in params:
                if isinstance(item, dict) and item.get("TRACE_ID"):
                    return item["TRACE_ID"]
        elif isinstance(params, dict):
            trace_ids = params.get("TRACE_ID", [])
            if trace_ids:
                return trace_ids[0] if isinstance(trace_ids, list) else trace_ids
    return None


def render_trace_detail(trace_id: str, tool_details: pd.DataFrame, spans: pd.DataFrame):
    trace_tools = tool_details[tool_details["TRACE_ID"] == trace_id] if not tool_details.empty else pd.DataFrame()
    trace_root = spans[(spans["TRACE_ID"] == trace_id) & (spans["SPAN_TYPE"] == "record_root")] if not spans.empty else pd.DataFrame()

    if not trace_root.empty:
        row = trace_root.iloc[0]
        q = row.get("INPUT_QUESTION", "")
        a = row.get("OUTPUT_ANSWER", "")
        if pd.notna(q) and q:
            st.markdown(f"**Question:** {q}")
        if pd.notna(a) and a:
            st.markdown(f"**Answer:** {str(a)[:1000]}")

    if trace_tools.empty:
        st.info(f"No tool/retrieval details for trace {trace_id}")
        return

    for _, row in trace_tools.iterrows():
        span_type = row["SPAN_TYPE"]
        if span_type == "tool":
            question = str(row["ANALYST_QUESTION"]) if pd.notna(row["ANALYST_QUESTION"]) else "N/A"
            interpretation = str(row["ANALYST_INTERPRETATION"]) if pd.notna(row.get("ANALYST_INTERPRETATION")) else ""
            generated_sql = str(row["ANALYST_GENERATED_SQL"]) if pd.notna(row.get("ANALYST_GENERATED_SQL")) else ""
            query_results = str(row["ANALYST_QUERY_RESULTS"]) if pd.notna(row.get("ANALYST_QUERY_RESULTS")) else ""
            call_return = str(row["CALL_RETURN"]) if pd.notna(row.get("CALL_RETURN")) else ""

            if not interpretation and not generated_sql and call_return:
                if "SQL:" in call_return and "Results:" in call_return:
                    parts = call_return.split("Results:", 1)
                    query_results = parts[1].strip() if len(parts) > 1 else ""
                    header_and_sql = parts[0]
                    if "SQL:" in header_and_sql:
                        sql_and_interp = header_and_sql.split("SQL:", 1)
                        interpretation = sql_and_interp[0].strip()
                        generated_sql = sql_and_interp[1].strip()
                elif "SQL:" in call_return:
                    sql_and_interp = call_return.split("SQL:", 1)
                    interpretation = sql_and_interp[0].strip()
                    generated_sql = sql_and_interp[1].strip()

            st.markdown(f"**{row['TOOL_FUNCTION']}** — {question[:80]}")
            st.caption(f"Latency: {row['LATENCY_MS']}ms")
            if interpretation:
                st.info(interpretation)
            if generated_sql:
                st.code(generated_sql.rstrip(";").strip(), language="sql")
            if query_results:
                st.text(query_results[:2000])
        elif span_type == "retrieval":
            query_text = str(row["RETRIEVAL_QUERY_TEXT"]) if pd.notna(row["RETRIEVAL_QUERY_TEXT"]) else (str(row["SEARCH_QUERY"]) if pd.notna(row["SEARCH_QUERY"]) else "N/A")
            contexts_raw = row["RETRIEVED_CONTEXTS_RAW"]
            contexts = []
            if contexts_raw:
                try:
                    contexts = json.loads(contexts_raw)
                except (json.JSONDecodeError, TypeError):
                    contexts = [str(contexts_raw)]

            st.markdown(f"#### Tool Call: `{row['TOOL_FUNCTION']}()`")
            col_q, col_lat = st.columns([4, 1])
            with col_q:
                st.markdown(f"> {query_text}")
            with col_lat:
                st.metric("Latency", f"{row['LATENCY_MS']}ms")
            if contexts:
                st.markdown(f"**{len(contexts)} retrieved chunk{'s' if len(contexts) != 1 else ''}**")
                for i, ctx in enumerate(contexts):
                    ctx_str = str(ctx) if not isinstance(ctx, str) else ctx
                    preview = ctx_str[:120].replace("\n", " ") + ("..." if len(ctx_str) > 120 else "")
                    with st.expander(f"Chunk {i+1} — {preview}", expanded=(i == 0)):
                        st.markdown(ctx_str[:2000])
            else:
                st.info("No retrieved contexts found.")
st.caption("Timeseries views of trace data, tool calls, eval scores, and latency from Snowflake AI Observability")

agents_df = load_agents()

if agents_df.empty:
    st.warning("No observability data found.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    agent_names = agents_df["AGENT_NAME"].tolist()
    default_agent = "SUPPORT CLOUD AGENT" if "SUPPORT CLOUD AGENT" in agent_names else agent_names[0]
    selected_agent = st.selectbox("Agent", agent_names, index=agent_names.index(default_agent))

runs_df = load_runs(selected_agent)

if runs_df.empty:
    st.warning(f"No runs found for {selected_agent}.")
    st.stop()

all_runs = runs_df["RUN_NAME"].tolist()
run_names_tuple = tuple(all_runs)
spans_df = load_spans(run_names_tuple)
tool_details_df = load_tool_details(run_names_tuple)
eval_roots_df = load_eval_roots(run_names_tuple)
server_evals_df = load_server_evals(run_names_tuple)

# --- KPI row ---
st.markdown("---")

record_roots = spans_df[spans_df["SPAN_TYPE"] == "record_root"] if not spans_df.empty else pd.DataFrame()
if not record_roots.empty:
    record_roots["LATENCY_MS"] = pd.to_numeric(record_roots["LATENCY_MS"], errors="coerce")
total_queries = len(record_roots)
avg_latency = float(record_roots["LATENCY_MS"].mean()) if not record_roots.empty else 0
total_tool_calls = len(spans_df[spans_df["SPAN_TYPE"].isin(["tool", "retrieval"])]) if not spans_df.empty else 0
total_gen_calls = len(spans_df[spans_df["SPAN_TYPE"] == "generation"]) if not spans_df.empty else 0

all_eval_scores = pd.DataFrame()
if not eval_roots_df.empty:
    valid_evals = eval_roots_df.dropna(subset=["SCORE"])
    all_eval_scores = valid_evals
if not server_evals_df.empty:
    valid_server = server_evals_df.dropna(subset=["SCORE"])
    if not valid_server.empty:
        server_as_eval = valid_server[["TIMESTAMP", "RUN_NAME", "METRIC_NAME", "SCORE"]]
        if not all_eval_scores.empty:
            client_cols = all_eval_scores[["TIMESTAMP", "RUN_NAME", "METRIC_NAME", "SCORE"]]
            all_eval_scores = pd.concat([client_cols, server_as_eval], ignore_index=True)
        else:
            all_eval_scores = server_as_eval

avg_eval_score = float(all_eval_scores["SCORE"].mean()) if not all_eval_scores.empty else 0

with st.container(horizontal=True):
    st.metric("Total Queries", f"{total_queries}", border=True)
    st.metric("Avg Latency", f"{avg_latency:,.0f} ms", border=True)
    st.metric("Tool / Retrieval Calls", f"{total_tool_calls}", border=True)
    st.metric("LLM Generations", f"{total_gen_calls}", border=True)
    st.metric("Avg Eval Score", f"{avg_eval_score:.2f}" if avg_eval_score else "N/A", border=True)

# --- Tabs ---
tab_latency, tab_tools, tab_evals, tab_traces = st.tabs([
    "Latency Over Time",
    "Tool Calls",
    "Eval Scores",
    "Trace Explorer",
])

# --- Latency Tab ---
with tab_latency:
    if record_roots.empty:
        st.info("No record_root spans found for selected runs.")
    else:
        st.subheader("Query Latency Over Time")
        latency_df = record_roots[["TIMESTAMP", "RUN_NAME", "LATENCY_MS", "INPUT_QUESTION", "TRACE_ID"]].copy()
        latency_df["TIMESTAMP"] = pd.to_datetime(latency_df["TIMESTAMP"])
        latency_df["LATENCY_MS"] = pd.to_numeric(latency_df["LATENCY_MS"], errors="coerce")

        q75 = latency_df["LATENCY_MS"].quantile(0.75)
        q25 = latency_df["LATENCY_MS"].quantile(0.25)
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        latency_df["IS_OUTLIER"] = latency_df["LATENCY_MS"] > outlier_threshold

        click = alt.selection_point(name="latency_select", fields=["TRACE_ID"])
        pan_zoom = alt.selection_interval(bind="scales")
        points = alt.Chart(latency_df).mark_circle().encode(
            x=alt.X("TIMESTAMP:T", title="Time"),
            y=alt.Y("LATENCY_MS:Q", title="Latency (ms)"),
            size=alt.condition(
                alt.datum.IS_OUTLIER,
                alt.value(120),
                alt.value(50),
            ),
            color=alt.condition(
                alt.datum.IS_OUTLIER,
                alt.value("#FF4B4B"),
                alt.value("#29B5E8"),
            ),
            tooltip=["INPUT_QUESTION", "LATENCY_MS", alt.Tooltip("TIMESTAMP:T", format="%Y-%m-%d %H:%M:%S"), "TRACE_ID"],
        ).add_params(click, pan_zoom)
        trend = alt.Chart(latency_df).mark_line(
            color="#FFA500", strokeDash=[6, 3], strokeWidth=2,
        ).transform_loess(
            "TIMESTAMP", "LATENCY_MS", bandwidth=0.4,
        ).encode(
            x="TIMESTAMP:T",
            y="LATENCY_MS:Q",
        )
        threshold_rule = alt.Chart(pd.DataFrame({"y": [outlier_threshold]})).mark_rule(
            color="#FF4B4B", strokeDash=[4, 4], strokeWidth=1,
        ).encode(y="y:Q")

        chart = (points + trend + threshold_rule).properties(height=400)
        latency_event = st.altair_chart(chart, key="latency_chart", on_select="rerun")
        st.caption(f"Orange = LOESS trend · Red dots = outliers (>{outlier_threshold:,.0f} ms) · Blue = normal · Click a dot to inspect trace · Drag to pan, scroll to zoom")

        selected_trace = _get_selected_trace_id(latency_event)
        if selected_trace:
            with st.container(border=True):
                st.subheader(f"Trace: {selected_trace}")
                render_trace_detail(selected_trace, tool_details_df, spans_df)

        st.subheader("Latency by Span Type")
        if not spans_df.empty:
            spans_for_latency = spans_df.copy()
            spans_for_latency["LATENCY_MS"] = pd.to_numeric(spans_for_latency["LATENCY_MS"], errors="coerce")
            span_latency = spans_for_latency.groupby("SPAN_TYPE").agg(
                avg_ms=("LATENCY_MS", "mean"),
                count=("LATENCY_MS", "count"),
            ).reset_index()
            span_latency.columns = ["Span Type", "Avg Latency (ms)", "Count"]
            span_latency["Avg Latency (ms)"] = span_latency["Avg Latency (ms)"].round(0).astype(int)

            bar = alt.Chart(span_latency).mark_bar().encode(
                x=alt.X("Span Type:N"),
                y=alt.Y("Avg Latency (ms):Q"),
                color="Span Type:N",
                tooltip=["Span Type", "Avg Latency (ms)", "Count"],
            ).properties(height=300)
            st.altair_chart(bar)

# --- Tool Calls Tab ---
with tab_tools:
    tool_spans = spans_df[spans_df["SPAN_TYPE"].isin(["tool", "retrieval"])].copy() if not spans_df.empty else pd.DataFrame()
    if not tool_spans.empty:
        tool_spans["LATENCY_MS"] = pd.to_numeric(tool_spans["LATENCY_MS"], errors="coerce")
    if tool_spans.empty:
        st.info("No tool/retrieval spans found.")
    else:
        st.subheader("Tool Call Latency Over Time")
        tool_ts = tool_spans[["TIMESTAMP", "RUN_NAME", "SPAN_TYPE", "TOOL_FUNCTION", "LATENCY_MS", "TRACE_ID"]].copy()
        tool_ts["TIMESTAMP"] = pd.to_datetime(tool_ts["TIMESTAMP"])

        click = alt.selection_point(name="tool_select", fields=["TRACE_ID"])
        pan_zoom = alt.selection_interval(bind="scales")
        points = alt.Chart(tool_ts).mark_circle(size=80).encode(
            x=alt.X("TIMESTAMP:T", title="Time"),
            y=alt.Y("LATENCY_MS:Q", title="Latency (ms)"),
            color=alt.Color("TOOL_FUNCTION:N", title="Function"),
            shape=alt.Shape("SPAN_TYPE:N", title="Type"),
            tooltip=["RUN_NAME", "TOOL_FUNCTION", "SPAN_TYPE", "LATENCY_MS", alt.Tooltip("TIMESTAMP:T", format="%Y-%m-%d %H:%M:%S"), "TRACE_ID"],
        ).add_params(click, pan_zoom)
        lines = alt.Chart(tool_ts).mark_line(
            opacity=0.5, strokeWidth=2,
        ).transform_loess(
            "TIMESTAMP", "LATENCY_MS", groupby=["TOOL_FUNCTION"], bandwidth=0.4,
        ).encode(
            x=alt.X("TIMESTAMP:T"),
            y=alt.Y("LATENCY_MS:Q"),
            color=alt.Color("TOOL_FUNCTION:N"),
        )
        chart = (lines + points).properties(height=400)
        tool_event = st.altair_chart(chart, key="tool_chart", on_select="rerun")
        st.caption("Click a dot to inspect trace · Drag to pan, scroll to zoom")

        selected_trace = _get_selected_trace_id(tool_event)
        if selected_trace:
            with st.container(border=True):
                st.subheader(f"Trace: {selected_trace}")
                render_trace_detail(selected_trace, tool_details_df, spans_df)

        st.subheader("Tool Call Distribution")
        col1, col2 = st.columns(2)
        with col1:
            tool_counts = tool_spans["TOOL_FUNCTION"].value_counts().reset_index()
            tool_counts.columns = ["Function", "Count"]
            pie = alt.Chart(tool_counts).mark_arc(innerRadius=50).encode(
                theta="Count:Q",
                color=alt.Color("Function:N"),
                tooltip=["Function", "Count"],
            ).properties(height=300)
            st.altair_chart(pie)

        with col2:
            tool_latency = tool_spans.groupby("TOOL_FUNCTION").agg(
                avg_ms=("LATENCY_MS", "mean"),
                count=("LATENCY_MS", "count"),
            ).reset_index()
            tool_latency.columns = ["Function", "Avg Latency (ms)", "Count"]
            tool_latency["Avg Latency (ms)"] = tool_latency["Avg Latency (ms)"].round(0).astype(int)
            st.dataframe(tool_latency, hide_index=True, use_container_width=True)



# --- Eval Scores Tab ---
with tab_evals:
    has_client = not eval_roots_df.empty
    has_server = not server_evals_df.empty

    if not has_client and not has_server:
        st.info("No evaluation data found for selected runs.")
    else:
        st.subheader("Eval Scores Over Time")

        record_id_to_trace = {}
        record_id_to_run = {}
        if not record_roots.empty:
            for _, rr in record_roots[["RECORD_ID", "TRACE_ID", "RUN_NAME"]].dropna(subset=["RECORD_ID"]).iterrows():
                record_id_to_trace[rr["RECORD_ID"]] = rr.get("TRACE_ID", "")
                record_id_to_run[rr["RECORD_ID"]] = rr.get("RUN_NAME", "")

        combined_evals = pd.DataFrame()
        if has_client:
            client = eval_roots_df[["TIMESTAMP", "RUN_NAME", "METRIC_NAME", "SCORE", "TARGET_RECORD_ID"]].copy()
            client["SOURCE"] = "client-side"
            combined_evals = client
        if has_server:
            server = server_evals_df[["TIMESTAMP", "RUN_NAME", "METRIC_NAME", "SCORE", "TARGET_RECORD_ID"]].copy()
            server["SOURCE"] = "server-side"
            combined_evals = pd.concat([combined_evals, server], ignore_index=True) if not combined_evals.empty else server

        combined_evals = combined_evals.dropna(subset=["SCORE"])
        combined_evals["SCORE"] = pd.to_numeric(combined_evals["SCORE"], errors="coerce")
        combined_evals = combined_evals.dropna(subset=["SCORE"])
        combined_evals["TIMESTAMP"] = pd.to_datetime(combined_evals["TIMESTAMP"])
        combined_evals["TRACE_ID"] = combined_evals["TARGET_RECORD_ID"].map(record_id_to_trace).fillna("")
        empty_run = combined_evals["RUN_NAME"].isna() | (combined_evals["RUN_NAME"] == "")
        combined_evals.loc[empty_run, "RUN_NAME"] = combined_evals.loc[empty_run, "TARGET_RECORD_ID"].map(record_id_to_run).fillna("unknown")

        if combined_evals.empty:
            st.info("All eval scores are NaN.")
        else:
            metrics = combined_evals["METRIC_NAME"].dropna().unique().tolist()
            selected_metrics = st.multiselect("Metrics to Display", metrics, default=metrics, key="eval_metrics")
            filtered = combined_evals[combined_evals["METRIC_NAME"].isin(selected_metrics)]

            click = alt.selection_point(name="eval_select", fields=["TRACE_ID"])
            pan_zoom = alt.selection_interval(bind="scales")
            points = alt.Chart(filtered).mark_circle(size=60).encode(
                x=alt.X("TIMESTAMP:T", title="Time"),
                y=alt.Y("SCORE:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("METRIC_NAME:N", title="Metric"),
                shape=alt.Shape("SOURCE:N", title="Source"),
                tooltip=["RUN_NAME", "METRIC_NAME", "SCORE", "SOURCE", alt.Tooltip("TIMESTAMP:T", format="%Y-%m-%d %H:%M:%S"), "TRACE_ID"],
            ).add_params(click, pan_zoom)
            lines = alt.Chart(filtered).mark_line(
                opacity=0.5, strokeWidth=2,
            ).transform_loess(
                "TIMESTAMP", "SCORE", groupby=["METRIC_NAME"], bandwidth=0.4,
            ).encode(
                x=alt.X("TIMESTAMP:T"),
                y=alt.Y("SCORE:Q"),
                color=alt.Color("METRIC_NAME:N"),
            )
            chart = (lines + points).properties(height=400)
            eval_event = st.altair_chart(chart, key="eval_chart", on_select="rerun")
            st.caption("Click a dot to inspect trace · Drag to pan, scroll to zoom")

            selected_trace = _get_selected_trace_id(eval_event)
            if selected_trace:
                with st.container(border=True):
                    st.subheader(f"Trace: {selected_trace}")
                    render_trace_detail(selected_trace, tool_details_df, spans_df)

            st.subheader("Score Distribution by Metric")
            box = alt.Chart(filtered).mark_boxplot(extent="min-max").encode(
                x=alt.X("METRIC_NAME:N", title="Metric"),
                y=alt.Y("SCORE:Q", title="Score"),
                color="METRIC_NAME:N",
            ).properties(height=300)
            st.altair_chart(box)

# --- Trace Explorer Tab ---
with tab_traces:
    if tool_details_df.empty:
        st.info("No tool/retrieval trace details found.")
    else:
        analyst_details = tool_details_df[tool_details_df["SPAN_TYPE"] == "tool"].dropna(subset=["CALL_RETURN"])
        retrieval_details = tool_details_df[tool_details_df["SPAN_TYPE"] == "retrieval"]

        if not analyst_details.empty:
            st.subheader("Analyst Call Details")
            for idx, row in analyst_details.iterrows():
                question = str(row["ANALYST_QUESTION"]) if pd.notna(row["ANALYST_QUESTION"]) else "N/A"
                interpretation = str(row["ANALYST_INTERPRETATION"]) if pd.notna(row.get("ANALYST_INTERPRETATION")) else ""
                generated_sql = str(row["ANALYST_GENERATED_SQL"]) if pd.notna(row.get("ANALYST_GENERATED_SQL")) else ""
                query_results = str(row["ANALYST_QUERY_RESULTS"]) if pd.notna(row.get("ANALYST_QUERY_RESULTS")) else ""

                if not interpretation and not generated_sql:
                    call_return = str(row["CALL_RETURN"]) if pd.notna(row["CALL_RETURN"]) else ""
                    if "SQL:" in call_return and "Results:" in call_return:
                        parts = call_return.split("Results:", 1)
                        query_results = parts[1].strip() if len(parts) > 1 else ""
                        header_and_sql = parts[0]
                        if "SQL:" in header_and_sql:
                            sql_and_interp = header_and_sql.split("SQL:", 1)
                            interpretation = sql_and_interp[0].strip()
                            generated_sql = sql_and_interp[1].strip()
                    elif "SQL:" in call_return:
                        sql_and_interp = call_return.split("SQL:", 1)
                        interpretation = sql_and_interp[0].strip()
                        generated_sql = sql_and_interp[1].strip()

                label = f"{row['TOOL_FUNCTION']} — {question[:80]}"
                with st.expander(label, expanded=False):
                    st.caption(f"Run: {row['RUN_NAME']} | Trace: {row['TRACE_ID']} | Latency: {row['LATENCY_MS']}ms")
                    st.markdown("**Question:**")
                    st.write(question)
                    if interpretation:
                        st.markdown("**Interpretation:**")
                        st.info(interpretation)
                    if generated_sql:
                        st.markdown("**Generated SQL:**")
                        st.code(generated_sql.rstrip(";").strip(), language="sql")
                    if query_results:
                        st.markdown("**Query Results:**")
                        st.text(query_results[:2000])

        if not retrieval_details.empty:
            st.subheader("Retrieval / Search Details")
            for idx, row in retrieval_details.iterrows():
                query_text = str(row["RETRIEVAL_QUERY_TEXT"]) if pd.notna(row["RETRIEVAL_QUERY_TEXT"]) else (str(row["SEARCH_QUERY"]) if pd.notna(row["SEARCH_QUERY"]) else "N/A")
                contexts_raw = row["RETRIEVED_CONTEXTS_RAW"]
                contexts = []
                if contexts_raw:
                    try:
                        contexts = json.loads(contexts_raw)
                    except (json.JSONDecodeError, TypeError):
                        contexts = [str(contexts_raw)]

                label = f"Tool Call: `{row['TOOL_FUNCTION']}()` — \"{query_text[:80]}\""
                with st.expander(label, expanded=False):
                    st.caption(f"Run: {row['RUN_NAME']} | Trace: {row['TRACE_ID']}")
                    col_q, col_lat = st.columns([4, 1])
                    with col_q:
                        st.markdown(f"**Search Query:**")
                        st.markdown(f"> {query_text}")
                    with col_lat:
                        st.metric("Latency", f"{row['LATENCY_MS']}ms")
                    if contexts:
                        st.markdown(f"**{len(contexts)} retrieved chunk{'s' if len(contexts) != 1 else ''}**")
                        for i, ctx in enumerate(contexts):
                            ctx_str = str(ctx) if not isinstance(ctx, str) else ctx
                            preview = ctx_str[:120].replace("\n", " ") + ("..." if len(ctx_str) > 120 else "")
                            with st.expander(f"Chunk {i+1} — {preview}", expanded=(i == 0)):
                                st.markdown(ctx_str[:2000])
                    else:
                        st.info("No retrieved contexts found.")
