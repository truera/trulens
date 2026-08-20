import codecs
import json
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.components.record_viewer_otel import record_viewer_otel
from trulens.dashboard.constants import EXTERNAL_APP_COL_NAME
from trulens.dashboard.constants import HIDE_RECORD_COL_NAME
from trulens.dashboard.constants import PINNED_COL_NAME
from trulens.dashboard.constants import RECORDS_PAGE_NAME as page_name
from trulens.dashboard.utils import dashboard_utils
from trulens.dashboard.utils import streamlit_compat
from trulens.dashboard.utils.dashboard_utils import ST_RECORDS_LIMIT
from trulens.dashboard.utils.dashboard_utils import _get_event_otel_spans
from trulens.dashboard.utils.dashboard_utils import _show_no_records_error
from trulens.dashboard.utils.dashboard_utils import get_feedback_defs
from trulens.dashboard.utils.dashboard_utils import get_records_and_feedback
from trulens.dashboard.utils.dashboard_utils import is_sis_compatibility_enabled
from trulens.dashboard.utils.dashboard_utils import (
    read_query_params_into_session_state,
)
from trulens.dashboard.utils.dashboard_utils import render_app_version_filters
from trulens.dashboard.utils.dashboard_utils import render_sidebar
from trulens.dashboard.utils.dashboard_utils import set_page_config
from trulens.dashboard.utils.records_utils import _render_feedback_call
from trulens.dashboard.utils.records_utils import _render_feedback_pills
from trulens.dashboard.utils.streamlit_compat import st_code
from trulens.dashboard.utils.streamlit_compat import st_columns
from trulens.dashboard.ux.styles import aggrid_css
from trulens.dashboard.ux.styles import cell_rules
from trulens.dashboard.ux.styles import default_direction
from trulens.dashboard.ux.styles import radio_button_css
from trulens.otel.semconv.trace import SpanAttributes


def init_page_state():
    if st.session_state.get(f"{page_name}.initialized", False):
        return

    read_query_params_into_session_state(
        page_name=page_name,
        transforms={
            "app_ids": lambda x: x.split(","),
            "app_versions": lambda x: x.split(","),
            "metric_name": lambda x: x.split(","),
            "selected_thread": lambda x: x,
        },
    )

    app_ids: Optional[List[str]] = st.session_state.get(
        f"{page_name}.app_ids", None
    )
    if app_ids and not st.query_params.get("app_ids", None):
        st.query_params["app_ids"] = ",".join(app_ids)

    st.session_state[f"{page_name}.initialized"] = True


def _format_cost(cost: float, currency: str) -> str:
    if currency == "USD":
        return f"${cost:.2f}"
    else:
        return f"{cost:.3g} {currency}"


def _clean_text_value(value: Any) -> Any:
    """Fix common text encoding issues and decode escaped sequences.

    - Fix UTF-8 displayed as Windows-1252/latin-1 (e.g., â -> ’)
    - Decode backslash-escaped unicode sequences when present
    - Leave non-strings unchanged
    """
    if not isinstance(value, str):
        return value

    text = value

    # Attempt to fix mojibake: bytes intended as UTF-8 shown as latin-1
    # Trigger only if typical mojibake markers present
    if any(mark in text for mark in ("â€", "Ã", "Â")):
        try:
            text = text.encode("latin-1", errors="ignore").decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            pass

    # Decode literal escape sequences like \u2019 -> ’ if present
    if "\\u" in text or "\\x" in text:
        try:
            text = codecs.decode(text.encode("utf-8"), "unicode_escape")
        except Exception:
            pass

    return text


def _apply_drilldown_sort(
    records_df: pd.DataFrame,
    metric_kind: Optional[str],
    metric_name: Optional[str],
    sort_direction: str,
    feedback_col_names: Sequence[str],
) -> tuple[pd.DataFrame, Optional[str]]:
    if not metric_kind:
        return records_df, None
    if metric_kind == "feedback":
        if metric_name not in feedback_col_names:
            return records_df, None
        sort_column = metric_name
    elif metric_kind == "latency":
        sort_column = "latency"
    elif metric_kind == "app_cost":
        sort_column = "total_cost"
    elif metric_kind == "eval_cost":
        sort_column = (
            "eval_cost_snowflake"
            if (
                "eval_cost_snowflake" in records_df.columns
                and (records_df["eval_cost_snowflake"] != 0).any()
            )
            else "eval_cost"
        )
    else:
        return records_df, None
    if sort_column not in records_df.columns:
        return records_df, None
    return (
        records_df.sort_values(
            sort_column,
            ascending=sort_direction == "asc",
            na_position="last",
        ),
        sort_column,
    )


def _escape_currency_dollars(text: str) -> str:
    """Escape $ so Markdown doesn't interpret it as LaTeX math.

    If we render with st.markdown, unescaped $ can trigger math in some themes.
    """
    if isinstance(text, str):
        return text.replace("$", r"\$")
    return text


def _escape_problematic_markdown(text: str) -> str:
    """Escape characters that can rewrite plain text when rendered as Markdown.

    - Escape dollar signs to avoid accidental math rendering
    - Escape underscores only when they are between word characters (e.g.,
      followed_by -> followed\_by). This preserves legitimate Markdown
      italics/bold like _text_ or __text__ which are typically surrounded by
      spaces or punctuation.
    """
    if not isinstance(text, str):
        return text
    escaped = text.replace("$", r"\$")
    # Escape underscores between alphanumerics
    import re as _re

    escaped = _re.sub(r"(?<=\w)_(?=\w)", r"\\_", escaped)
    return escaped


def _get_record_ttft_ms(selected_row: pd.Series) -> Optional[float]:
    """Time to first token (ms) for this record, if it contains a streaming
    GENERATION span. `None` if OTel tracing is off, the record has no such
    span, or the span didn't capture a TTFT (e.g. it never got a first
    chunk). When a record has multiple streaming generation calls, this
    returns the first one found rather than an aggregate -- attribute
    naming makes it explicit which record it came from either way."""
    if not is_otel_tracing_enabled():
        return None
    try:
        event_spans = _get_event_otel_spans(
            selected_row["record_id"], selected_row["app_name"]
        )
    except Exception:
        return None
    for span in event_spans:
        attrs = span.get("record_attributes") or {}
        if attrs.get(SpanAttributes.GENERATION.IS_STREAMING):
            ttft = attrs.get(SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS)
            if ttft is not None:
                return ttft
    return None


def _render_record_metrics(
    records_df: pd.DataFrame, selected_row: pd.Series
) -> None:
    """Render record level metrics (e.g. total tokens, cost, latency) compared
    to the average when appropriate."""

    app_specific_df = records_df[records_df["app_id"] == selected_row["app_id"]]

    ttft_ms = _get_record_ttft_ms(selected_row)
    if ttft_ms is not None:
        token_col, cost_col, latency_col, ttft_col, _ = st_columns([1, 1, 1, 1, 2])
    else:
        token_col, cost_col, latency_col, _ = st_columns([1, 1, 1, 3])

    num_tokens = selected_row["total_tokens"]
    with token_col.container(height=128, border=True):
        st.metric(
            label="Total tokens (#)",
            value=num_tokens,
            help="Number of tokens generated for this record.",
        )

    cost = selected_row["total_cost"]
    cost_currency = selected_row["cost_currency"]
    average_cost = app_specific_df[
        app_specific_df["cost_currency"] == cost_currency
    ]["total_cost"].mean()
    delta_cost = cost - average_cost

    with cost_col.container(height=128, border=True):
        st.metric(
            label=f"Total cost ({cost_currency})",
            value=_format_cost(cost, cost_currency),
            delta=f"{delta_cost:.3g} {cost_currency}"
            if delta_cost != 0
            else None,
            delta_color="inverse",
            help=f"Cost of the app execution measured in {cost_currency}. Delta is relative to average cost for the app."
            if delta_cost != 0
            else f"Cost of the app execution measured in {cost_currency}.",
        )

    latency = selected_row["latency"]
    average_latency = app_specific_df["latency"].mean()
    delta_latency = latency - average_latency
    with latency_col.container(height=128, border=True):
        st.metric(
            label="Latency (s)",
            value=f"{selected_row['latency']}s",
            delta=f"{delta_latency:.3g}s" if delta_latency != 0 else None,
            delta_color="inverse",
            help="Latency of the app execution. Delta is relative to average latency for the app."
            if delta_latency != 0
            else "Latency of the app execution.",
        )

    if ttft_ms is not None:
        with ttft_col.container(height=128, border=True):
            st.metric(
                label="Time to First Token (ms)",
                value=f"{ttft_ms:.0f}ms",
                help="Time between issuing the request and receiving the "
                "first streamed chunk, for this record's (first) "
                "streaming generation call.",
            )


def _render_record_detail(
    selected_row: pd.Series,
    records_df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    key_suffix: Optional[str] = None,
):
    """Render the full detail view for a single record.

    This is the canonical record display (input/output, metrics, feedback
    results, and trace details). It is reused both for standalone records and
    for each turn within a conversation thread. ``key_suffix`` makes the
    widget/component keys unique when several records are rendered on the same
    page (e.g. one per turn in a thread).
    """
    suffix = key_suffix if key_suffix is not None else selected_row["record_id"]

    # Start the record specific section
    st.divider()

    # Breadcrumbs
    st.caption(
        f"{selected_row['app_name']} / {selected_row['app_version']} / Record: {selected_row['record_id']}"
    )
    st.markdown(f"#### Record ID: {selected_row['record_id']}")

    input_col, output_col = st_columns(2)
    with input_col.expander("Record Input"):
        input_value = selected_row["input"]
        if isinstance(input_value, str):
            text = _clean_text_value(input_value)
            text = _escape_problematic_markdown(text)
            st.markdown(text, unsafe_allow_html=False)
        else:
            st_code(selected_row["input"], wrap_lines=True)

    with output_col.expander("Record Output"):
        output_value = selected_row["output"]
        if isinstance(output_value, str):
            text = _clean_text_value(output_value)
            text = _escape_problematic_markdown(text)
            st.markdown(text, unsafe_allow_html=False)
        else:
            st_code(json.dumps(output_value, indent=2), wrap_lines=True)

    _render_record_metrics(records_df, selected_row)

    online_eval_status = selected_row.get("online_eval_status")
    if online_eval_status:
        sample_rate = selected_row.get("sample_rate_display", "—")
        st.caption(
            f"Online evaluation: **{online_eval_status}** · "
            f"Sample rate: **{sample_rate}**"
        )

    app_json = selected_row["app_json"]
    record_json = selected_row["record_json"]

    feedback_results = st.container()
    trace_details = st.container()

    # Feedback results
    feedback_results.subheader("Feedback Results")
    with feedback_results:
        if selected_ff := _render_feedback_pills(
            feedback_col_names=feedback_col_names,
            selected_row=selected_row,
            feedback_directions=feedback_directions,
            key=f"{page_name}.feedback_pills.{suffix}",
        ):
            _render_feedback_call(
                selected_ff,
                selected_row,
                feedback_directions=feedback_directions,
            )

    # Trace details

    if is_sis_compatibility_enabled():
        st.subheader("Trace Details")
        st.json(record_json, expanded=1)

        st.subheader("App Details")
        st.json(app_json, expanded=1)
    elif is_otel_tracing_enabled():
        with trace_details:
            st.subheader("Trace Details")
            event_spans = _get_event_otel_spans(
                selected_row["record_id"], selected_row["app_name"]
            )
            if event_spans:
                record_viewer_otel(
                    spans=event_spans,
                    key=f"{page_name}.trace.{suffix}",
                )
            else:
                st.warning("No trace data available for this record.")
    else:
        with trace_details:
            st.subheader("Trace Details")
            record_viewer(
                record_json,
                app_json,
                key=f"{page_name}.trace_legacy.{suffix}",
            )


@streamlit_compat.st_fragment
def _render_trace(
    selected_row: pd.Series,
    records_df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
):
    _render_record_detail(
        selected_row,
        records_df,
        feedback_col_names,
        feedback_directions,
    )


def _preprocess_df(
    records_df: pd.DataFrame,
    record_query: Optional[str] = None,
    online_eval_filter: str = "All",
):
    records_df = dashboard_utils.add_online_eval_columns(records_df)
    if HIDE_RECORD_COL_NAME in records_df.columns:
        records_df = records_df[~records_df[HIDE_RECORD_COL_NAME]]
    records_df = records_df.sort_values(by="ts", ascending=False)
    if "input" in records_df.columns:
        records_df["input"] = records_df["input"].apply(_clean_text_value)
    if "output" in records_df.columns:
        records_df["output"] = records_df["output"].apply(_clean_text_value)

    # Reorder App columns to the front. Used for aggrid display
    records_df.insert(0, "app_id", records_df.pop("app_id"))
    records_df.insert(0, "app_name", records_df.pop("app_name"))
    records_df.insert(0, "record_id", records_df.pop("record_id"))
    records_df.insert(0, "app_version", records_df.pop("app_version"))

    # if "app_json" in records_df.columns:
    #     records_df = records_df.drop(columns="app_json")

    if record_query:
        records_df = records_df[
            records_df["app_version"].str.contains(record_query, case=False)
            | records_df["input"].str.contains(record_query, case=False)
            | records_df["output"].str.contains(record_query, case=False)
        ]
    if online_eval_filter == "Selected":
        records_df = records_df[records_df["online_eval_status"] == "Selected"]
    elif online_eval_filter == "Skipped":
        records_df = records_df[
            records_df["online_eval_status"].str.startswith("Skipped")
        ]
    elif online_eval_filter == "Not configured":
        records_df = records_df[
            records_df["online_eval_status"] == "Not configured"
        ]
    records_df["ts"] = pd.to_datetime(records_df["ts"])
    return records_df


def _build_grid_options(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: Sequence[str],
):
    from st_aggrid.grid_options_builder import GridOptionsBuilder

    gb = GridOptionsBuilder.from_dataframe(df, headerHeight=50)
    gb.configure_default_column(resizable=True, flex=0, suppressSizeToFit=True)

    gb.configure_column(
        "app_version",
        header_name="App Version",
        resizable=True,
        pinned="left",
        flex=3,
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "app_name",
        header_name="App Name",
        hide=True,
        filter="agMultiColumnFilter",
    )

    gb.configure_column(
        PINNED_COL_NAME,
        header_name="Pinned",
        hide=True,
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        EXTERNAL_APP_COL_NAME,
        header_name="External App",
        hide=True,
        filter="agSetColumnFilter",
    )

    gb.configure_column(
        "app_id", header_name="App ID", hide=True, filter="agSetColumnFilter"
    )
    gb.configure_column(
        "record_id",
        header_name="Record ID",
        pinned="left",
        hide=True,
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        "online_eval_status",
        header_name="Online Eval",
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        "sample_rate_display",
        header_name="Sample Rate",
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        "eval_decision_reason",
        header_name="Decision",
        filter="agSetColumnFilter",
    )

    gb.configure_column(
        "input",
        header_name="Record Input",
        cellStyle={
            "white-space": "nowrap",
            "overflow": "hidden",
            "text-overflow": "ellipsis",
        },
        width=300,
        minWidth=300,
        maxWidth=300,
        resizable=False,
        suppressSizeToFit=True,
        tooltipField="input",
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "output",
        header_name="Record Output",
        cellStyle={
            "white-space": "nowrap",
            "overflow": "hidden",
            "text-overflow": "ellipsis",
        },
        width=300,
        minWidth=300,
        maxWidth=300,
        resizable=False,
        suppressSizeToFit=True,
        tooltipField="output",
        filter="agMultiColumnFilter",
    )

    gb.configure_column(
        "record_metadata",
        header_name="Record Metadata",
        filter="agTextColumnFilter",
        valueGetter="JSON.stringify(data.record_metadata)",
    )

    gb.configure_column(
        "total_tokens",
        header_name="Total Tokens (#)",
        filter="agNumberColumnFilter",
    )
    gb.configure_column(
        "total_cost",
        header_name="Total Cost (App)",
        filter="agNumberColumnFilter",
    )
    # Dynamically include all nonzero eval cost columns and hide zero/missing ones
    has_eval_sf = (
        "eval_cost_snowflake" in df.columns
        and (df["eval_cost_snowflake"] != 0).any()
    )
    has_eval_usd = "eval_cost" in df.columns and (df["eval_cost"] != 0).any()

    # Show all nonzero eval cost columns (both if applicable); hide zero/missing
    if has_eval_sf:
        gb.configure_column(
            "eval_cost_snowflake",
            header_name="Eval Costs (Snowflake Credits)",
            filter="agNumberColumnFilter",
        )
    elif "eval_cost_snowflake" in df.columns:
        gb.configure_column("eval_cost_snowflake", hide=True)

    if has_eval_usd:
        gb.configure_column(
            "eval_cost",
            header_name="Eval Costs",
            filter="agNumberColumnFilter",
        )
    elif "eval_cost" in df.columns:
        gb.configure_column("eval_cost", hide=True)
    gb.configure_column(
        "cost_currency",
        header_name="Cost Currency",
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "latency",
        header_name="Latency (Seconds)",
        filter="agNumberColumnFilter",
    )
    gb.configure_column(
        "tags", header_name="Application Tag", filter="agSetColumnFilter"
    )
    gb.configure_column(
        "ts", header_name="Time Stamp", sort="desc", filter="agDateColumnFilter"
    )

    gb.configure_column(
        "feedback_id",
        header_name="Feedback ID",
        hide=True,
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "type", header_name="App Type", hide=True, filter="agMultiColumnFilter"
    )
    gb.configure_column(
        "record_json",
        header_name="Record JSON",
        hide=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "app_json",
        header_name="App JSON",
        hide=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "cost_json",
        header_name="Cost JSON",
        hide=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "perf_json",
        header_name="Perf. JSON",
        hide=True,
        filter="agTextColumnFilter",
    )
    gb.configure_column(
        "num_events",
        header_name="Number of Events",
    )

    for metadata_col in version_metadata_col_names:
        gb.configure_column(
            metadata_col,
            header_name=metadata_col,
            resizable=True,
            pinned=True,
            hide=True,
        )

    for feedback_col in feedback_col_names:
        if "distance" in feedback_col:
            gb.configure_column(feedback_col, hide=False)
            gb.configure_column(feedback_col + "_calls", hide=True)
        else:
            feedback_direction = (
                "HIGHER_IS_BETTER"
                if feedback_directions.get(feedback_col, default_direction)
                else "LOWER_IS_BETTER"
            )
            gb.configure_column(
                feedback_col,
                cellClassRules=cell_rules[feedback_direction],
                hide=False,
            )
            gb.configure_column(
                feedback_col + "_calls",
                hide=True,
            )
            gb.configure_column(
                feedback_col + " direction",
                cellDataType="text",
                valueGetter=f"data['{feedback_col} direction'] ? 'Higher is Better' : 'Lower is Better'",
                hide=True,
            )

    for col in df.columns:
        if "feedback cost" in col:
            gb.configure_column(
                col,
                hide=True,
            )
    gb.configure_grid_options(
        rowHeight=40,
        suppressContextMenu=True,
        suppressSizeToFit=True,
    )
    gb.configure_selection(
        selection_mode="single",
        use_checkbox=True,
    )
    gb.configure_pagination(enabled=True)
    gb.configure_side_bar()

    return gb.build()


def _render_grid(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: Sequence[str],
):
    if not is_sis_compatibility_enabled():
        try:
            import st_aggrid
            from st_aggrid.shared import ColumnsAutoSizeMode
            from st_aggrid.shared import DataReturnMode

            event = st_aggrid.AgGrid(
                df,
                gridOptions=_build_grid_options(
                    df=df,
                    feedback_col_names=feedback_col_names,
                    feedback_directions=feedback_directions,
                    version_metadata_col_names=version_metadata_col_names,
                ),
                update_on=["selectionChanged"],
                custom_css={**aggrid_css, **radio_button_css},
                columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,
                fit_columns_on_grid_load=False,
                reload_data=True,
                data_return_mode=DataReturnMode.FILTERED,
                allow_unsafe_jscode=True,
            )
            return pd.DataFrame(event.selected_rows)

        except ImportError:
            # Fallback to st.dataframe if st_aggrid is not installed
            pass

    # Build column order dynamically for eval cost columns
    eval_cols_to_show = []
    has_eval_sf = (
        "eval_cost_snowflake" in df.columns
        and (df["eval_cost_snowflake"] != 0).any()
    )
    has_eval_usd = "eval_cost" in df.columns and (df["eval_cost"] != 0).any()
    if has_eval_sf:
        eval_cols_to_show.append("eval_cost_snowflake")
    if has_eval_usd:
        eval_cols_to_show.append("eval_cost")

    column_order = [
        "app_version",
        "online_eval_status",
        "sample_rate_display",
        "eval_decision_reason",
        "input",
        "output",
        "record_metadata",
        "total_tokens",
        "total_cost",
        *eval_cols_to_show,
        "cost_currency",
        "latency",
        "tags",
        "ts",
        *version_metadata_col_names,
        *feedback_col_names,
        "record_json",
    ]
    column_order = [col for col in column_order if col in df.columns]
    event = st.dataframe(
        df[column_order],
        column_order=column_order,
        selection_mode="single-row",
        on_select="rerun",
        hide_index=True,
        width="stretch",
    )
    return df.iloc[event.selection["rows"]]


def _render_grid_tab(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    version_metadata_col_names: List[str],
):
    selected_records = _render_grid(
        df,
        feedback_col_names=feedback_col_names,
        feedback_directions=feedback_directions,
        version_metadata_col_names=version_metadata_col_names,
    )

    if selected_records.empty:
        selected_record_id = st.session_state.get(
            f"{page_name}.selected_record", None
        )
        selected_records = df[df["record_id"] == selected_record_id]

    if selected_records.empty:
        st.info(
            "Click a record's checkbox to view details.",
            icon="ℹ️",
        )
        return

    selected_record = selected_records.iloc[0]
    st.session_state[f"{page_name}.selected_record"] = selected_record[
        "record_id"
    ]
    st.query_params["selected_record"] = selected_record["record_id"]
    _render_trace(selected_record, df, feedback_col_names, feedback_directions)


def _has_conversations(df: pd.DataFrame) -> bool:
    """Return True if any record in the dataframe carries a conversation_id."""
    if "conversation_id" not in df.columns:
        return False
    conv = df["conversation_id"]
    return bool((conv.notna() & (conv.astype(str).str.len() > 0)).any())


def _thread_membership(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that belong to a conversation/thread."""
    if "conversation_id" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    conv = df["conversation_id"]
    return conv.notna() & (conv.astype(str).str.len() > 0)


def _thread_keys(df: pd.DataFrame) -> pd.Series:
    is_thread = _thread_membership(df)
    keys = df["record_id"].map(lambda value: f"record:{value}")
    keys.loc[is_thread] = df.loc[is_thread].apply(
        lambda row: json.dumps(
            [row["app_name"], row["app_version"], row["conversation_id"]],
            separators=(",", ":"),
        ),
        axis=1,
    )
    return keys


def _build_thread_summary(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    sort_col: Optional[str] = None,
    sort_direction: str = "desc",
) -> pd.DataFrame:
    """Collapse the records dataframe into one row per thread.

    Records that share a ``conversation_id`` are grouped into a single thread.
    Records without a ``conversation_id`` become standalone single-message
    "threads" keyed by their ``record_id`` so they still appear and open
    normally.
    """
    df = df.copy()
    is_thread = _thread_membership(df)
    df["_thread_key"] = _thread_keys(df)
    df["_is_thread"] = is_thread
    if "is_match" not in df.columns:
        df["is_match"] = True

    rows = []
    for thread_key, group in df.groupby("_thread_key", dropna=False):
        group_sorted = group.sort_values("ts")
        first = group_sorted.iloc[0]
        last = group_sorted.iloc[-1]
        is_conv = bool(first["_is_thread"])
        matched = group_sorted[group_sorted["is_match"].fillna(False)]
        ranking_group = matched if not matched.empty else group_sorted
        row: Dict[str, Any] = {
            "thread_key": thread_key,
            "is_thread": is_conv,
            "conversation_id": first["conversation_id"] if is_conv else None,
            "record_id": None if is_conv else first["record_id"],
            "app_name": first["app_name"],
            "app_version": first["app_version"],
            "app_id": first["app_id"],
            "num_messages": len(group_sorted),
            "matched_turn_count": len(matched),
            "total_turn_count": len(group_sorted),
            "first_input": first["input"],
            "last_output": last["output"],
            "start_ts": first["ts"],
            "ts": last["ts"],
            "total_tokens": group_sorted["total_tokens"].sum(),
            "total_cost": group_sorted["total_cost"].sum(),
            "cost_currency": first["cost_currency"],
            "latency": group_sorted["latency"].sum(),
        }
        if sort_col and sort_col in ranking_group.columns:
            values = pd.to_numeric(
                ranking_group[sort_col], errors="coerce"
            ).dropna()
            if not values.empty:
                worst_index = (
                    values.idxmin()
                    if sort_direction == "asc"
                    else values.idxmax()
                )
                row["thread_sort_value"] = values.loc[worst_index]
                row["worst_matching_record_id"] = ranking_group.loc[
                    worst_index, "record_id"
                ]
        for fcol in feedback_col_names:
            if fcol in ranking_group.columns:
                row[fcol] = ranking_group[fcol].mean(skipna=True)
        rows.append(row)

    thread_df = pd.DataFrame(rows)
    if not thread_df.empty:
        if "thread_sort_value" in thread_df.columns:
            thread_df = thread_df.sort_values(
                by=["thread_sort_value", "ts"],
                ascending=[sort_direction == "asc", False],
                na_position="last",
            )
        else:
            thread_df = thread_df.sort_values(by="ts", ascending=False)
    return thread_df


def _build_thread_grid_options(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
):
    from st_aggrid.grid_options_builder import GridOptionsBuilder

    gb = GridOptionsBuilder.from_dataframe(df, headerHeight=50)
    gb.configure_default_column(resizable=True, flex=0, suppressSizeToFit=True)

    gb.configure_column("thread_key", hide=True)
    gb.configure_column("is_thread", hide=True)
    gb.configure_column("app_name", hide=True)
    gb.configure_column("app_id", hide=True)
    gb.configure_column("start_ts", hide=True)
    gb.configure_column("cost_currency", hide=True)
    gb.configure_column("thread_sort_value", hide=True)
    gb.configure_column("worst_matching_record_id", hide=True)
    gb.configure_column(
        "conversation_id",
        header_name="Conversation ID",
        pinned="left",
        flex=2,
        filter="agSetColumnFilter",
        checkboxSelection=True,
    )
    gb.configure_column(
        "record_id",
        header_name="Record ID",
        pinned="left",
        flex=2,
        filter="agSetColumnFilter",
    )
    gb.configure_column(
        "app_version",
        header_name="App Version",
        pinned="left",
        flex=2,
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "num_messages",
        header_name="Messages (#)",
        filter="agNumberColumnFilter",
    )
    gb.configure_column(
        "matched_turn_count",
        header_name="Relevant turns (#)",
        filter="agNumberColumnFilter",
    )
    gb.configure_column("total_turn_count", hide=True)
    gb.configure_column(
        "first_input",
        header_name="First Input",
        cellStyle={
            "white-space": "nowrap",
            "overflow": "hidden",
            "text-overflow": "ellipsis",
        },
        width=300,
        minWidth=300,
        maxWidth=300,
        resizable=False,
        suppressSizeToFit=True,
        tooltipField="first_input",
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "last_output",
        header_name="Last Output",
        cellStyle={
            "white-space": "nowrap",
            "overflow": "hidden",
            "text-overflow": "ellipsis",
        },
        width=300,
        minWidth=300,
        maxWidth=300,
        resizable=False,
        suppressSizeToFit=True,
        tooltipField="last_output",
        filter="agMultiColumnFilter",
    )
    gb.configure_column(
        "total_tokens",
        header_name="Total Tokens (#)",
        filter="agNumberColumnFilter",
    )
    gb.configure_column(
        "total_cost", header_name="Total Cost", filter="agNumberColumnFilter"
    )
    gb.configure_column(
        "latency", header_name="Latency (s)", filter="agNumberColumnFilter"
    )
    gb.configure_column(
        "ts",
        header_name="Last Activity",
        sort="desc",
        filter="agDateColumnFilter",
    )

    for fcol in feedback_col_names:
        if fcol not in df.columns:
            continue
        feedback_direction = (
            "HIGHER_IS_BETTER"
            if feedback_directions.get(fcol, default_direction)
            else "LOWER_IS_BETTER"
        )
        gb.configure_column(
            fcol,
            header_name=f"{fcol} (avg)",
            cellClassRules=cell_rules[feedback_direction],
            hide=False,
        )

    gb.configure_grid_options(
        rowHeight=40,
        suppressContextMenu=True,
        suppressSizeToFit=True,
    )
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_pagination(enabled=True)
    gb.configure_side_bar()
    return gb.build()


def _render_thread_grid(
    df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
):
    if not is_sis_compatibility_enabled():
        try:
            import st_aggrid
            from st_aggrid.shared import ColumnsAutoSizeMode
            from st_aggrid.shared import DataReturnMode

            event = st_aggrid.AgGrid(
                df,
                gridOptions=_build_thread_grid_options(
                    df=df,
                    feedback_col_names=feedback_col_names,
                    feedback_directions=feedback_directions,
                ),
                update_on=["selectionChanged"],
                custom_css={**aggrid_css, **radio_button_css},
                columns_auto_size_mode=ColumnsAutoSizeMode.NO_AUTOSIZE,
                fit_columns_on_grid_load=False,
                reload_data=True,
                data_return_mode=DataReturnMode.FILTERED,
                allow_unsafe_jscode=True,
                key=f"{page_name}.thread_grid",
            )
            return pd.DataFrame(event.selected_rows)
        except ImportError:
            pass

    column_order = [
        "conversation_id",
        "record_id",
        "app_version",
        "matched_turn_count",
        "num_messages",
        "first_input",
        "last_output",
        "total_tokens",
        "total_cost",
        "latency",
        "ts",
        *[f for f in feedback_col_names if f in df.columns],
    ]
    column_order = [col for col in column_order if col in df.columns]
    event = st.dataframe(
        df[column_order],
        column_order=column_order,
        selection_mode="single-row",
        on_select="rerun",
        hide_index=True,
        width="stretch",
        key=f"{page_name}.thread_grid_df",
    )
    return df.iloc[event.selection["rows"]]


def _clear_selected_thread():
    if f"{page_name}.selected_thread" in st.session_state:
        del st.session_state[f"{page_name}.selected_thread"]
    if "selected_thread" in st.query_params:
        st.query_params.pop("selected_thread")


def _conversation_turn_html(
    *,
    record_id: str,
    turn_label: str,
    border: str,
    user_text: str,
    assistant_text: str,
) -> str:
    return (
        f'<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:8px;border-left:3px solid {border};padding-left:8px;">'
        f'  <span style="color:{border};font-size:0.72em;font-weight:600;">{turn_label}</span>'
        f'  <div style="display:flex;align-items:flex-start;">'
        f'    <div style="background:var(--secondary-background-color);color:var(--text-color);border:1px solid color-mix(in srgb, var(--text-color) 12%, transparent);border-radius:12px 12px 12px 2px;padding:8px 12px;max-width:75%;font-size:0.85em;">'
        f'      <span style="color:color-mix(in srgb, var(--text-color) 65%, transparent);font-size:0.75em;">User</span><br>{user_text}'
        f"    </div>"
        f"  </div>"
        f'  <div style="display:flex;align-items:flex-start;justify-content:flex-end;">'
        f'    <div style="background:color-mix(in srgb, var(--primary-color) 14%, var(--background-color));color:var(--text-color);border:1px solid color-mix(in srgb, var(--primary-color) 35%, transparent);border-radius:12px 12px 2px 12px;padding:8px 12px;max-width:75%;font-size:0.85em;">'
        f'      <span style="color:color-mix(in srgb, var(--text-color) 65%, transparent);font-size:0.75em;">Assistant</span><br>{assistant_text}'
        f"    </div>"
        f'    <a href="#record-{record_id}" style="margin-left:8px;margin-top:8px;text-decoration:none;font-size:0.75em;color:var(--text-color);opacity:0.65;white-space:nowrap;">trace&nbsp;↓</a>'
        f"  </div>"
        f"</div>"
    )


def _partition_feedback_scopes(
    subset: pd.DataFrame,
    feedback_col_names: Sequence[str],
) -> tuple[list[str], list[str]]:
    conversation_metrics = []
    turn_metrics = []
    for metric_name in feedback_col_names:
        calls_column = f"{metric_name}_calls"
        is_conversation_metric = False
        if calls_column in subset.columns:
            for calls in subset[calls_column].dropna():
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    args_span_attribute = call.get("args_span_attribute", {})
                    if isinstance(args_span_attribute, dict) and (
                        "conversation.records" in args_span_attribute.values()
                    ):
                        is_conversation_metric = True
                        break
                if is_conversation_metric:
                    break
        if is_conversation_metric:
            conversation_metrics.append(metric_name)
        else:
            turn_metrics.append(metric_name)
    return conversation_metrics, turn_metrics


def _conversation_metric_row(
    subset: pd.DataFrame, metric_name: str
) -> Optional[pd.Series]:
    rows = subset[subset[metric_name].notna()]
    if rows.empty:
        return None
    return rows.sort_values("ts").iloc[-1]


def _split_conversation_turns(
    subset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(subset) <= 2:
        return subset, subset.iloc[0:0], subset.iloc[0:0]
    return subset.iloc[[0]], subset.iloc[1:-1], subset.iloc[[-1]]


def _render_conversation_turns(subset: pd.DataFrame) -> None:
    for _, msg in subset.iterrows():
        record_id = msg["record_id"]
        is_match = bool(msg.get("is_match", True))
        turn_label = "Conversation turn"
        border = (
            "var(--primary-color)"
            if is_match
            else "color-mix(in srgb, var(--text-color) 35%, transparent)"
        )
        input_text = str(msg["input"])[:200]
        output_text = str(msg["output"])[:200]
        ellipsis_in = "..." if len(str(msg["input"])) > 200 else ""
        ellipsis_out = "..." if len(str(msg["output"])) > 200 else ""
        st.html(
            _conversation_turn_html(
                record_id=record_id,
                turn_label=turn_label,
                border=border,
                user_text=_escape_problematic_markdown(input_text)
                + ellipsis_in,
                assistant_text=_escape_problematic_markdown(output_text)
                + ellipsis_out,
            )
        )


def _render_conversation_timeline(subset: pd.DataFrame):
    """Render first/last turns with collapsed middle conversation turns."""
    first_turns, middle_turns, last_turns = _split_conversation_turns(subset)
    _render_conversation_turns(first_turns)
    if not middle_turns.empty:
        with st.expander(
            f"See {len(middle_turns)} more turns",
            expanded=False,
        ):
            _render_conversation_turns(middle_turns)
    _render_conversation_turns(last_turns)


def _render_thread(
    thread_key: str,
    records_df: pd.DataFrame,
    feedback_col_names: Sequence[str],
    feedback_directions: Dict[str, bool],
):
    """Render a single thread as a sequence of its messages.

    Each message (turn) is rendered with the exact same detail view used for a
    standalone record (``_render_record_detail``), so a conversation is just a
    vertical stack of record detail views in turn order.
    """
    subset = records_df[_thread_keys(records_df) == thread_key]
    if subset.empty:
        return

    subset = subset.sort_values(by="ts", ascending=True)
    first = subset.iloc[0]

    st.divider()
    st.button(
        "← Back to threads",
        on_click=_clear_selected_thread,
        key=f"{page_name}.back_to_threads",
    )

    # Standalone (non-conversation) records keep the existing record detail view.
    if not bool(first["conversation_id"]):
        _render_trace(
            first, records_df, feedback_col_names, feedback_directions
        )
        return

    conversation_id = first["conversation_id"]
    match_flags = subset.get("is_match", pd.Series(True, index=subset.index))
    matched_count = int(match_flags.fillna(False).sum())
    st.caption(
        f"{first['app_name']} / {first['app_version']} / Thread: {conversation_id}"
    )
    st.markdown(f"#### Thread: {conversation_id}")
    st.caption(
        f"{matched_count} relevant turn(s) · {len(subset)} total turn(s) in this conversation"
    )

    # Conversation-level stats
    cost_currency = first["cost_currency"]
    total_cost = subset[subset["cost_currency"] == cost_currency][
        "total_cost"
    ].sum()
    total_time = subset["latency"].sum()
    total_tokens = int(subset["total_tokens"].sum())

    # Left: chat-style timeline. Right: conversation-level stats.
    timeline_col, stats_col = st.columns(2)
    with timeline_col:
        _render_conversation_timeline(subset)
    with stats_col:
        stat_top = st.columns(2)
        stat_bottom = st.columns(2)
        with stat_top[0].container(height=128, border=True):
            st.metric(
                label="Turns (#)",
                value=len(subset),
                help="Number of messages (records) in this conversation.",
            )
        with stat_top[1].container(height=128, border=True):
            st.metric(
                label="Total time (s)",
                value=f"{total_time:.3g}",
                help="Sum of per-turn latency across the conversation.",
            )
        with stat_bottom[0].container(height=128, border=True):
            st.metric(
                label=f"Total cost ({cost_currency})",
                value=_format_cost(total_cost, cost_currency),
                help="Sum of app execution cost across all turns (excludes eval cost).",
            )
        with stat_bottom[1].container(height=128, border=True):
            st.metric(
                label="Total AI tokens (#)",
                value=f"{total_tokens:,}",
                help="Sum of tokens generated across all turns.",
            )

    available_metrics = [
        f
        for f in feedback_col_names
        if f in subset.columns and subset[f].notna().any()
    ]
    conversation_metrics, turn_metrics = _partition_feedback_scopes(
        subset, available_metrics
    )

    if conversation_metrics:
        st.markdown("#### Conversation metrics")
        conversation_scores = pd.Series({
            metric_name: _conversation_metric_row(subset, metric_name)[
                metric_name
            ]
            for metric_name in conversation_metrics
        })
        selected_metric = _render_feedback_pills(
            conversation_metrics,
            feedback_directions,
            selected_row=conversation_scores,
            key=f"{page_name}.conversation_metrics.{thread_key}",
        )
        if selected_metric:
            metric_row = _conversation_metric_row(subset, selected_metric)
            if metric_row is not None:
                _render_feedback_call(
                    selected_metric,
                    metric_row,
                    feedback_directions=feedback_directions,
                )

    if turn_metrics:
        st.markdown("#### Turn metrics")
        metric_subset = subset[match_flags.fillna(False)]
        if metric_subset.empty:
            metric_subset = subset
        avg_row = pd.Series({
            fcol: metric_subset[fcol].mean(skipna=True) for fcol in turn_metrics
        })
        pills_key = f"{page_name}.thread_avg_pills.{thread_key}"
        selected_metric = _render_feedback_pills(
            turn_metrics,
            feedback_directions,
            selected_row=avg_row,
            key=pills_key,
        )
        if selected_metric and selected_metric in subset.columns:
            columns = ["record_id", "input", selected_metric]
            turn_scores = subset[columns].copy()
            turn_scores.insert(0, "Turn", range(1, len(turn_scores) + 1))
            turn_scores["input"] = turn_scores["input"].astype(str).str[:80]
            turn_scores = turn_scores.rename(
                columns={
                    "record_id": "Record ID",
                    "input": "Input",
                    selected_metric: f"{selected_metric} (score)",
                }
            )
            st.dataframe(
                turn_scores.set_index("Turn"),
                use_container_width=True,
            )

    for turn_index, (_, msg) in enumerate(subset.iterrows()):
        record_id = msg["record_id"]
        st.html(
            f'<div id="record-{record_id}" style="height:0;margin:0;padding:0;overflow:hidden;"></div>'
        )
        st.caption(
            "Conversation turn"
            if bool(msg.get("is_match", True))
            else "Conversation context turn"
        )
        _render_record_detail(
            msg,
            records_df,
            turn_metrics,
            feedback_directions,
            key_suffix=f"{thread_key}.{turn_index}.{msg['record_id']}",
        )


def _render_thread_view(
    df: pd.DataFrame,
    feedback_col_names: List[str],
    feedback_directions: Dict[str, bool],
    sort_col: Optional[str] = None,
    sort_direction: str = "desc",
):
    """Render the thread-grouped records view with drill-in transcript."""
    thread_summary = _build_thread_summary(
        df, feedback_col_names, sort_col, sort_direction
    )
    if thread_summary.empty:
        st.info("No threads to display.", icon="ℹ️")
        return

    selected = _render_thread_grid(
        thread_summary,
        feedback_col_names=feedback_col_names,
        feedback_directions=feedback_directions,
    )

    valid_keys = set(thread_summary["thread_key"].tolist())
    if selected is not None and not selected.empty:
        selected_thread = selected.iloc[0]["thread_key"]
        st.session_state[f"{page_name}.selected_thread"] = selected_thread
        st.query_params["selected_thread"] = str(selected_thread)
    else:
        selected_thread = st.session_state.get(
            f"{page_name}.selected_thread", None
        )

    if selected_thread is not None and selected_thread in valid_keys:
        _render_thread(
            selected_thread, df, feedback_col_names, feedback_directions
        )
    else:
        st.info(
            "Click a thread's checkbox to view the conversation.",
            icon="ℹ️",
        )


def _reset_app_ids():
    if f"{page_name}.app_ids" in st.session_state:
        del st.session_state[f"{page_name}.app_ids"]
    if "app_ids" in st.query_params:
        st.query_params.pop("app_ids")


def _render_app_id_args_filter(versions_df: pd.DataFrame):
    if app_ids := st.session_state.get(f"{page_name}.app_ids", None):
        # Reading session state from other pages
        valid_app_ids = versions_df["app_id"].unique()
        if not set(app_ids).issubset(valid_app_ids):
            ids_str = "**`" + "`**, **`".join(app_ids) + "`**"
            st.error(f"Got invalid App IDs: {ids_str}")
            _reset_app_ids()
            return versions_df

        ids_str = "**`" + "`**, **`".join(app_ids) + "`**"
        info_col, show_all_col = st_columns(
            [0.9, 0.1], vertical_alignment="center"
        )
        info_col.info(f"Filtering with App IDs: {ids_str}")
        versions_df = versions_df[versions_df["app_id"].isin(app_ids)]

        show_all_col.button(
            "Show All", width="stretch", on_click=_reset_app_ids
        )
        if not st.query_params.get("app_ids", None):
            st.query_params["app_ids"] = ",".join(app_ids)
    return versions_df


def _handle_record_query_change():
    value = st.session_state.get(f"{page_name}.record_search", None)
    if value:
        st.query_params["record_search"] = value
    elif "record_search" in st.query_params:
        st.query_params.pop("record_search")


def _render_drilldown_summary(sort_column: Optional[str]) -> None:
    metric_kind = st.session_state.get(f"{page_name}.metric_kind")
    if not metric_kind:
        return
    metric_names = st.session_state.get(
        f"{page_name}.metric_name", [metric_kind]
    )
    if isinstance(metric_names, str):
        metric_names = metric_names.split(",")
    app_versions = st.session_state.get(f"{page_name}.app_versions", [])
    start_time = pd.Timestamp(st.session_state[f"{page_name}.start_time"])
    end_time = pd.Timestamp(st.session_state[f"{page_name}.end_time"])
    direction = st.session_state.get(f"{page_name}.sort_direction", "desc")
    with st.container(border=True):
        cols = st.columns([5, 1], vertical_alignment="center")
        cols[0].markdown(
            f"**Investigating:** {', '.join(metric_names)} · "
            f"{', '.join(app_versions)} · "
            f"{start_time:%b %d}–{(end_time - pd.Timedelta(days=1)):%b %d}"
        )
        if sort_column:
            cols[0].caption(
                f"Sorted by {sort_column}, "
                + ("lowest first." if direction == "asc" else "highest first.")
            )
        if cols[1].button("Clear investigation", width="stretch"):
            for name in (
                "app_versions",
                "metric_kind",
                "metric_name",
                "start_time",
                "end_time",
                "sort_direction",
                "currency",
            ):
                st.session_state.pop(f"{page_name}.{name}", None)
                st.query_params.pop(name, None)
            st.rerun()


def render_records(app_name: str):
    """Renders the records page.

    Args:
        app_name (str): The name of the app to render records for.
    """
    st.title(page_name)
    st.markdown(f"Showing app `{app_name}`")

    # Get app versions
    st.session_state.setdefault(f"{page_name}.record_search", "")
    controls = st_columns([2, 1], vertical_alignment="bottom")
    record_query = controls[0].text_input(
        "Search Records",
        key=f"{page_name}.record_search",
        on_change=_handle_record_query_change,
    )
    online_eval_filter = controls[1].selectbox(
        "Evaluation decision",
        ["All", "Selected", "Skipped", "Not configured"],
        key=f"{page_name}.online_eval_filter",
    )
    versions_df, _ = render_app_version_filters(
        app_name,
        {f"{page_name}.record_search": record_query},
        page_name_keys=[f"{page_name}.record_search"],
    )

    st.divider()

    versions_df = _render_app_id_args_filter(versions_df)

    if versions_df.empty:
        st.error(f"No app versions found for app `{app_name}`.")
        return
    app_versions = versions_df["app_version"].tolist()
    drilldown_versions = st.session_state.get(f"{page_name}.app_versions")
    if drilldown_versions:
        app_versions = [
            version for version in app_versions if version in drilldown_versions
        ]

    start_time_value = st.session_state.get(f"{page_name}.start_time")
    end_time_value = st.session_state.get(f"{page_name}.end_time")
    start_time = pd.Timestamp(start_time_value) if start_time_value else None
    end_time = pd.Timestamp(end_time_value) if end_time_value else None
    metric_kind = st.session_state.get(f"{page_name}.metric_kind")
    metric_names = st.session_state.get(f"{page_name}.metric_name", [])
    if isinstance(metric_names, str):
        metric_names = metric_names.split(",")
    drilldown_record_ids = None
    if metric_kind in ("feedback", "eval_cost") and start_time is not None:
        drilldown_record_ids = dashboard_utils.get_eval_drilldown_record_ids(
            app_name=app_name,
            app_versions=app_versions,
            metric_kind=metric_kind,
            metric_name=",".join(metric_names),
            start_time=start_time,
            end_time=end_time,
            currency=st.session_state.get(f"{page_name}.currency"),
        )

    # Get records and feedback data
    records_limit = st.session_state.get(ST_RECORDS_LIMIT, None)
    records_df, feedback_col_names = get_records_and_feedback(
        app_name=app_name,
        app_versions=app_versions,
        matched_record_ids=drilldown_record_ids,
        include_conversation_context=drilldown_record_ids is not None,
        start_time=start_time if drilldown_record_ids is None else None,
        end_time=end_time if drilldown_record_ids is None else None,
        limit=records_limit,
    )

    feedback_col_names = list(feedback_col_names)
    records_df, drilldown_sort_col = _apply_drilldown_sort(
        records_df,
        st.session_state.get(f"{page_name}.metric_kind"),
        metric_names[0] if metric_names else None,
        st.session_state.get(f"{page_name}.sort_direction", "desc"),
        feedback_col_names,
    )

    # Preprocess data
    df = _preprocess_df(
        records_df,
        record_query=record_query,
        online_eval_filter=online_eval_filter,
    )
    _render_drilldown_summary(drilldown_sort_col)

    if df.empty:
        if record_query:
            st.error(f"No records found for search query `{record_query}`.")
        elif online_eval_filter != "All":
            st.error(
                "No records found for evaluation decision "
                f"`{online_eval_filter}`."
            )
        elif app_ids := st.session_state.get(f"{page_name}.app_ids", None):
            app_versions = versions_df[versions_df["app_id"].isin(app_ids)][
                "app_version"
            ].unique()
            versions_str = "**`" + "`**, **`".join(app_versions) + "`**"
            st.error(f"No records found for app version(s): {versions_str}.")
        else:
            # Check for cross-format records before showing generic error
            _show_no_records_error(app_name=app_name, app_versions=app_versions)
        return
    elif records_limit is not None and len(records_df) >= records_limit:
        cols = st_columns([0.9, 0.1], vertical_alignment="center")
        cols[0].info(
            f"Limiting to the latest {records_limit} records. Use the search bar and filters to narrow your search.",
            icon="ℹ️",
        )

        def handle_show_all():
            st.session_state[ST_RECORDS_LIMIT] = None
            if ST_RECORDS_LIMIT in st.query_params:
                del st.query_params[ST_RECORDS_LIMIT]

        cols[1].button(
            "Show all",
            width="stretch",
            on_click=handle_show_all,
            help="Show all records. This may take a while.",
        )

    else:
        st.success(f"Found {len(df)} records.")

    _, feedback_directions = get_feedback_defs()

    _render_thread_view(
        df,
        feedback_col_names,
        feedback_directions,
        drilldown_sort_col,
        st.session_state.get(f"{page_name}.sort_direction", "desc"),
    )


def records_main():
    set_page_config(page_title=page_name)
    init_page_state()
    app_name = render_sidebar()
    if app_name:
        render_records(app_name)


if __name__ == "__main__":
    records_main()
