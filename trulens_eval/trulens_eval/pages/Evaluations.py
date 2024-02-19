import asyncio
import json
from typing import Iterable, Tuple

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

from pprint import PrettyPrinter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from ux.add_logo import add_logo_and_style_overrides
from ux.styles import CATEGORY

pp = PrettyPrinter()

from trulens_eval import Tru
from trulens_eval.app import Agent
from trulens_eval.app import ComponentView
from trulens_eval.app import instrumented_component_views
from trulens_eval.app import LLM
from trulens_eval.app import Other
from trulens_eval.app import Prompt
from trulens_eval.app import Tool
from trulens_eval.db import MULTI_CALL_NAME_DELIMITER
from trulens_eval.react_components.record_viewer import record_viewer
from trulens_eval.schema import Record
from trulens_eval.schema import Select
from trulens_eval.utils.json import jsonify_for_ui
from trulens_eval.utils.serial import Lens
from trulens_eval.ux.components import draw_agent_info
from trulens_eval.ux.components import draw_call
from trulens_eval.ux.components import draw_llm_info
from trulens_eval.ux.components import draw_metadata
from trulens_eval.ux.components import draw_prompt_info
from trulens_eval.ux.components import draw_tool_info
from trulens_eval.ux.components import render_selector_markdown
from trulens_eval.ux.components import write_or_json
from trulens_eval.ux.styles import cellstyle_jscode

st.set_page_config(page_title="Evaluations", layout="wide")

st.title("Evaluations")

st.runtime.legacy_caching.clear_cache()

add_logo_and_style_overrides()

tru = Tru()
lms = tru.db

df_results, feedback_cols = lms.get_records_and_feedback([])

# TODO: remove code redundancy / redundant database calls
feedback_directions = {
    (
        row.feedback_json.get("supplied_name", "") or row.feedback_json["implementation"]["name"]
    ):
        (
            "HIGHER_IS_BETTER"
            if row.feedback_json.get("higher_is_better", True) else
            "LOWER_IS_BETTER"
        ) for _, row in lms.get_feedback_defs().iterrows()
}
default_direction = "HIGHER_IS_BETTER"


def render_component(query, component, header=True):
    # Draw the accessor/path within the wrapped app of the component.
    if header:
        st.markdown(
            f"##### Component {render_selector_markdown(Select.for_app(query))}"
        )

    # Draw the python class information of this component.
    cls = component.cls
    base_cls = cls.base_class()
    label = f"__{repr(cls)}__"
    if str(base_cls) != str(cls):
        label += f" < __{repr(base_cls)}__"
    st.write("Python class: " + label)

    # Per-component-type drawing routines.
    if isinstance(component, LLM):
        draw_llm_info(component=component, query=query)

    elif isinstance(component, Prompt):
        draw_prompt_info(component=component, query=query)

    elif isinstance(component, Agent):
        draw_agent_info(component=component, query=query)

    elif isinstance(component, Tool):
        draw_tool_info(component=component, query=query)

    elif isinstance(component, Other):
        with st.expander("Uncategorized Component Details:"):
            st.json(jsonify_for_ui(component.json))

    else:
        with st.expander("Unhandled Component Details:"):
            st.json(jsonify_for_ui(component.json))


# Renders record level metrics (e.g. total tokens, cost, latency) compared to the average when appropriate
def render_record_metrics(app_df: pd.DataFrame, selected_rows: pd.DataFrame):
    app_specific_df = app_df[app_df["app_id"] == selected_rows["app_id"][0]]

    token_col, cost_col, latency_col = st.columns(3)

    num_tokens = selected_rows["total_tokens"][0]
    token_col.metric(label="Total tokens (#)", value=num_tokens)

    cost = selected_rows["total_cost"][0]
    average_cost = app_specific_df["total_cost"].mean()
    delta_cost = "{:.3g}".format(cost - average_cost)
    cost_col.metric(
        label="Total cost (USD)",
        value=selected_rows["total_cost"][0],
        delta=delta_cost,
        delta_color="inverse",
    )

    latency = selected_rows["latency"][0]
    average_latency = app_specific_df["latency"].mean()
    delta_latency = "{:.3g}s".format(latency - average_latency)
    latency_col.metric(
        label="Latency (s)",
        value=selected_rows["latency"][0],
        delta=delta_latency,
        delta_color="inverse",
    )


# Define a function to extract record metadata from each row
def extract_metadata(row):
    """
    Extract metadata from the record_json and return the metadata as a string.

    Args:
        row: The row containing the record_json.

    Returns:
        str: The metadata extracted from the record_json.
    """
    record_data = json.loads(row['record_json'])
    return str(record_data["meta"])


if df_results.empty:
    st.write("No records yet...")

else:
    apps = list(df_results.app_id.unique())
    if "app" in st.session_state:
        app = st.session_state.app
    else:
        app = apps

    st.query_params['app'] = app

    options = st.multiselect("Filter Applications", apps, default=app)

    if len(options) == 0:
        st.header("All Applications")
        app_df = df_results

    elif len(options) == 1:
        st.header(options[0])

        app_df = df_results[df_results.app_id.isin(options)]

    else:
        st.header("Multiple Applications Selected")

        app_df = df_results[df_results.app_id.isin(options)]

    tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

    with tab1:
        gridOptions = {"alwaysShowHorizontalScroll": True}
        evaluations_df = app_df

        # By default the cells in the df are unicode-escaped, so we have to reverse it.
        input_array = evaluations_df['input'].to_numpy()
        output_array = evaluations_df['output'].to_numpy()

        decoded_input = np.vectorize(
            lambda x: x.encode('utf-8').decode('unicode-escape')
        )(input_array)
        decoded_output = np.vectorize(
            lambda x: x.encode('utf-8').decode('unicode-escape')
        )(output_array)

        evaluations_df['input'] = decoded_input
        evaluations_df['output'] = decoded_output

        # Apply the function to each row and create a new column 'record_metadata'
        evaluations_df['record_metadata'] = evaluations_df.apply(
            extract_metadata, axis=1
        )

        gb = GridOptionsBuilder.from_dataframe(evaluations_df)

        gb.configure_column("type", header_name="App Type")
        gb.configure_column("record_json", header_name="Record JSON", hide=True)
        gb.configure_column("app_json", header_name="App JSON", hide=True)
        gb.configure_column("cost_json", header_name="Cost JSON", hide=True)
        gb.configure_column("perf_json", header_name="Perf. JSON", hide=True)

        gb.configure_column("record_id", header_name="Record ID", hide=True)
        gb.configure_column("app_id", header_name="App ID")

        gb.configure_column("feedback_id", header_name="Feedback ID", hide=True)
        gb.configure_column("input", header_name="User Input")
        gb.configure_column("output", header_name="Response")
        gb.configure_column("record_metadata", header_name="Record Metadata")

        gb.configure_column("total_tokens", header_name="Total Tokens (#)")
        gb.configure_column("total_cost", header_name="Total Cost (USD)")
        gb.configure_column("latency", header_name="Latency (Seconds)")
        gb.configure_column("tags", header_name="Application Tag")
        gb.configure_column("ts", header_name="Time Stamp", sort="desc")

        non_feedback_cols = [
            "app_id",
            "type",
            "ts",
            "total_tokens",
            "total_cost",
            "record_json",
            "latency",
            "tags",
            "record_metadata",
            "record_id",
            "cost_json",
            "app_json",
            "input",
            "output",
            "perf_json",
        ]

        for feedback_col in evaluations_df.columns.drop(non_feedback_cols):
            if "distance" in feedback_col:
                gb.configure_column(
                    feedback_col, hide=feedback_col.endswith("_calls")
                )
            else:
                # cell highlight depending on feedback direction
                cellstyle = JsCode(
                    cellstyle_jscode[feedback_directions.get(
                        feedback_col, default_direction
                    )]
                )

                gb.configure_column(
                    feedback_col,
                    cellStyle=cellstyle,
                    hide=feedback_col.endswith("_calls")
                )

        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        # gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        data = AgGrid(
            evaluations_df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
        )

        selected_rows = data["selected_rows"]
        selected_rows = pd.DataFrame(selected_rows)

        if len(selected_rows) == 0:
            st.write("Hint: select a row to display details of a record")

        else:
            # Start the record specific section
            st.divider()

            # Breadcrumbs
            st.caption(
                f"{selected_rows['app_id'][0]} / {selected_rows['record_id'][0]}"
            )
            st.header(f"{selected_rows['record_id'][0]}")

            render_record_metrics(app_df, selected_rows)

            st.markdown("")

            prompt = selected_rows["input"][0]
            response = selected_rows["output"][0]
            details = selected_rows["app_json"][0]
            record_json = selected_rows["record_json"][0]
            record_metadata = selected_rows["record_metadata"][0]

            app_json = json.loads(
                details
            )  # apps may not be deserializable, don't try to, keep it json.

            row = selected_rows.head().iloc[0]

            # Display input/response side by side. In each column, we put them in tabs mainly for
            # formatting/styling purposes.
            input_col, response_col = st.columns(2)

            (input_tab,) = input_col.tabs(["Input"])
            with input_tab:
                with st.expander(
                        f"Input {render_selector_markdown(Select.RecordInput)}",
                        expanded=True):
                    write_or_json(st, obj=prompt)

            (response_tab,) = response_col.tabs(["Response"])
            with response_tab:
                with st.expander(
                        f"Response {render_selector_markdown(Select.RecordOutput)}",
                        expanded=True):
                    write_or_json(st, obj=response)

            feedback_tab, metadata_tab = st.tabs(["Feedback", "Metadata"])

            with metadata_tab:
                metadata_dict = json.loads(record_json).get("meta", None)
                if metadata_dict is None:
                    st.write("No record metadata available")
                elif not isinstance(metadata_dict, dict):
                    st.write("Invalid metadata format: expected a dictionary (dict) type")
                else:
                    metadata_cols = list(metadata_dict.keys())

                    metadata_cols = st.columns(len(metadata_cols))

                    for i, (key, value) in enumerate(metadata_dict.items()):
                        metadata_cols[i].metric(
                            label=key,
                            value=value,
                        )

            with feedback_tab:
                if len(feedback_cols) == 0:
                    st.write("No feedback details")

                for fcol in feedback_cols:
                    feedback_name = fcol
                    feedback_result = row[fcol]

                    if MULTI_CALL_NAME_DELIMITER in fcol:
                        fcol = fcol.split(MULTI_CALL_NAME_DELIMITER)[0]
                    feedback_calls = row[f"{fcol}_calls"]

                    def display_feedback_call(call):

                        def highlight(s):
                            if "distance" in feedback_name:
                                return [
                                    f"background-color: {CATEGORY.UNKNOWN.color}"
                                ] * len(s)
                            cat = CATEGORY.of_score(
                                s.result,
                                higher_is_better=feedback_directions.get(
                                    fcol, default_direction
                                ) == default_direction
                            )
                            return [f"background-color: {cat.color}"] * len(s)

                        if call is not None and len(call) > 0:
                            # NOTE(piotrm for garett): converting feedback
                            # function inputs to strings here as other
                            # structures get rendered as [object Object] in the
                            # javascript downstream. If the first input/column
                            # is a list, the DataFrame.from_records does create
                            # multiple rows, one for each element, but if the
                            # second or other column is a list, it will not do
                            # this.
                            for c in call:
                                args = c['args']
                                for k, v in args.items():
                                    if not isinstance(v, str):
                                        args[k] = pp.pformat(v)

                            df = pd.DataFrame.from_records(
                                c['args'] for c in call
                            )

                            df["result"] = pd.DataFrame(
                                [
                                    float(call[i]["ret"])
                                    if call[i]["ret"] is not None else -1
                                    for i in range(len(call))
                                ]
                            )
                            df["meta"] = pd.Series(
                                [call[i]["meta"] for i in range(len(call))]
                            )
                            df = df.join(df.meta.apply(lambda m: pd.Series(m))
                                        ).drop(columns="meta")

                            st.dataframe(
                                df.style.apply(highlight, axis=1).format(
                                    "{:.2f}", subset=["result"]
                                )
                            )

                        else:
                            st.text("No feedback details.")

                    with st.expander(f"{feedback_name} = {feedback_result}",
                                     expanded=True):
                        display_feedback_call(feedback_calls)

            record_str = selected_rows["record_json"][0]
            record_json = json.loads(record_str)
            record = Record.model_validate(record_json)

            classes: Iterable[Tuple[Lens, ComponentView]
                             ] = list(instrumented_component_views(app_json))
            classes_map = {path: view for path, view in classes}

            st.markdown("")
            st.subheader("Timeline")
            val = record_viewer(record_json, app_json)
            st.markdown("")

            match_query = None

            # Assumes record_json['perf']['start_time'] is always present
            if val != "":
                match = None
                for call in record.calls:
                    if call.perf.start_time.isoformat() == val:
                        match = call
                        break

                if match:
                    length = len(match.stack)
                    app_call = match.stack[length - 1]

                    match_query = match.top().path

                    st.subheader(
                        f"{app_call.method.obj.cls.name} {render_selector_markdown(Select.for_app(match_query))}"
                    )

                    draw_call(match)

                    view = classes_map.get(match_query)
                    if view is not None:
                        render_component(
                            query=match_query, component=view, header=False
                        )
                    else:
                        st.write(
                            f"Call by `{match_query}` was not associated with any instrumented"
                            " component."
                        )
                        # Look up whether there was any data at that path even if not an instrumented component:

                        try:
                            app_component_json = list(
                                match_query.get(app_json)
                            )[0]
                            if app_component_json is not None:
                                with st.expander(
                                        "Uninstrumented app component details."
                                ):
                                    st.json(app_component_json)
                        except Exception:
                            st.write(
                                f"Recorded invocation by component `{match_query}` but cannot find this component in the app json."
                            )

                else:
                    st.text("No match found")
            else:
                st.subheader(f"App {render_selector_markdown(Select.App)}")
                with st.expander("App Details:"):
                    st.json(jsonify_for_ui(app_json))

            if match_query is not None:
                container = st.empty()

                has_subcomponents = False
                for query, component in classes:
                    if not match_query.is_immediate_prefix_of(query):
                        continue

                    if len(query.path) == 0:
                        # Skip App, will still list App.app under "app".
                        continue

                    has_subcomponents = True
                    render_component(query, component)

                if has_subcomponents:
                    container.markdown("#### Subcomponents:")

            st.header("More options:")

            if st.button("Display full app json"):
                st.write(jsonify_for_ui(app_json))

            if st.button("Display full record json"):
                st.write(jsonify_for_ui(record_json))

    with tab2:
        feedback = feedback_cols
        cols = 4
        rows = len(feedback) // cols + 1

        for row_num in range(rows):
            with st.container():
                columns = st.columns(cols)
                for col_num in range(cols):
                    with columns[col_num]:
                        ind = row_num * cols + col_num
                        if ind < len(feedback):
                            # Generate histogram
                            fig, ax = plt.subplots()
                            bins = [
                                0, 0.2, 0.4, 0.6, 0.8, 1.0
                            ]  # Quintile buckets
                            ax.hist(
                                app_df[feedback[ind]],
                                bins=bins,
                                edgecolor="black",
                                color="#2D736D"
                            )
                            ax.set_xlabel("Feedback Value")
                            ax.set_ylabel("Frequency")
                            ax.set_title(feedback[ind], loc="center")
                            st.pyplot(fig)
