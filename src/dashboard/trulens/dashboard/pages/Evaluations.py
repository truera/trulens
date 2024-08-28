import asyncio
import json
import pprint as pp
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from streamlit_pills import pills
from trulens.core import TruSession
from trulens.core.app import LLM
from trulens.core.app import Agent
from trulens.core.app import ComponentView
from trulens.core.app import Other
from trulens.core.app import Prompt
from trulens.core.app import Tool
from trulens.core.database.base import MULTI_CALL_NAME_DELIMITER
from trulens.core.schema.select import Select
from trulens.core.utils.json import jsonify_for_ui
from trulens.core.utils.serial import Lens
from trulens.dashboard.components.record_viewer import record_viewer
from trulens.dashboard.streamlit_utils import init_from_args
from trulens.dashboard.ux.components import draw_agent_info
from trulens.dashboard.ux.components import draw_llm_info
from trulens.dashboard.ux.components import draw_prompt_info
from trulens.dashboard.ux.components import draw_tool_info
from trulens.dashboard.ux.components import render_selector_markdown
from trulens.dashboard.ux.page_config import set_page_config
from trulens.dashboard.ux.styles import CATEGORY
from trulens.dashboard.ux.styles import cellstyle_jscode

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

set_page_config(page_title="Evaluations")
st.title("Evaluations")

if __name__ == "__main__":
    # If not imported, gets args from command line and creates Tru singleton
    init_from_args()

session = TruSession()
lms = session.connector.db

# TODO: remove code redundancy / redundant database calls
feedback_directions = {
    (
        row.feedback_json.get("supplied_name", "")
        or row.feedback_json["implementation"]["name"]
    ): (
        "HIGHER_IS_BETTER"
        if row.feedback_json.get("higher_is_better", True)
        else "LOWER_IS_BETTER"
    )
    for _, row in lms.get_feedback_defs().iterrows()
}
default_direction = "HIGHER_IS_BETTER"


def render_component(
    query: Lens, component: ComponentView, header: bool = True
) -> None:
    """Render the accessor/path within the wrapped app of the component."""

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


def render_record_metrics(
    app_df: pd.DataFrame, selected_rows: pd.DataFrame
) -> None:
    """Render record level metrics (e.g. total tokens, cost, latency) compared
    to the average when appropriate."""

    app_specific_df = app_df[app_df["app_id"] == selected_rows["app_id"][0]]

    token_col, cost_col, latency_col = st.columns(3)

    num_tokens = selected_rows["total_tokens"][0]
    token_col.metric(label="Total tokens (#)", value=num_tokens)

    cost = selected_rows["total_cost"][0]
    average_cost = app_specific_df["total_cost"].mean()
    delta_cost = f"{cost - average_cost:.3g}"
    cost_col.metric(
        label="Total cost (USD)",
        value=selected_rows["total_cost"][0],
        delta=delta_cost,
        delta_color="inverse",
    )

    latency = selected_rows["latency"][0]
    average_latency = app_specific_df["latency"].mean()
    delta_latency = f"{latency - average_latency:.3g}s"
    latency_col.metric(
        label="Latency (s)",
        value=selected_rows["latency"][0],
        delta=delta_latency,
        delta_color="inverse",
    )


def extract_metadata(row: pd.Series) -> str:
    """Extract metadata from the record_json and return the metadata as a string.

    Args:
        row: The row containing the record_json.

    Returns:
        str: The metadata extracted from the record_json.
    """
    record_data = json.loads(row["record_json"])
    return str(record_data["meta"])


app_ids = list(app["app_id"] for app in lms.get_apps())


def get_apps():
    return list(lms.get_apps())


apps = get_apps()

if "app" in st.session_state:
    app = st.session_state.app
else:
    app = app_ids

st.query_params["app"] = app

app_name_versions = list(
    set(f"{app['app_name']} - {app['app_version']}" for app in apps)
)

if "app" in st.session_state:
    selected_apps = st.session_state.app
    filtered_apps = [app for app in apps if app["app_id"] in selected_apps]
    app_name_versions = list(
        set(
            f"{app['app_name']} - {app['app_version']}" for app in filtered_apps
        )
    )
selected_app_name_versions = st.multiselect(
    "Filter Apps", app_name_versions, default=app_name_versions
)

selected_apps = [
    app["app_id"]
    for app in apps
    if f"{app['app_name']} - {app['app_version']}" in selected_app_name_versions
]

df_results, feedback_cols = lms.get_records_and_feedback(app_ids=selected_apps)

if len(selected_apps) == 0:
    st.header("All Applications")

elif len(selected_apps) == 1:
    st.header(selected_app_name_versions[0])

else:
    st.header("Multiple Applications Selected")

if df_results.empty:
    st.write("No records yet...")
else:
    app_df = df_results

    tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

    with tab1:
        gridOptions = {"alwaysShowHorizontalScroll": True}
        evaluations_df = app_df

        # By default the cells in the df are unicode-escaped, so we have to reverse it.
        input_array = evaluations_df["input"].to_numpy()
        output_array = evaluations_df["output"].to_numpy()

        # pull additional data from app_json into columns
        evaluations_df["app_name"] = evaluations_df["app_json"].apply(
            lambda x: json.loads(x)["app_name"]
        )
        evaluations_df["app_version"] = evaluations_df["app_json"].apply(
            lambda x: json.loads(x)["app_version"]
        )

        metadata = json.loads(evaluations_df["app_json"][0])["metadata"]
        metadata_cols = list(metadata.keys())
        for key in metadata_cols:
            evaluations_df[key] = evaluations_df["app_json"].apply(
                lambda x: json.loads(x)["metadata"].get(key)
            )

        # Drop the original app_metadata column if it exists
        if "app_metadata" in evaluations_df.columns:
            evaluations_df.drop(columns=["app_metadata"], inplace=True)

        # Reorder columns
        columns = (
            ["app_name", "app_version", "tags"]
            + metadata_cols
            + [
                col
                for col in evaluations_df.columns
                if col
                not in (["app_name", "app_version", "tags"] + metadata_cols)
            ]
        )
        evaluations_df = evaluations_df[columns]

        decoded_input = np.vectorize(
            lambda x: x.encode("utf-8").decode("unicode-escape")
        )(input_array)
        decoded_output = np.vectorize(
            lambda x: x.encode("utf-8").decode("unicode-escape")
        )(output_array)

        evaluations_df["input"] = decoded_input
        evaluations_df["output"] = decoded_output

        # Apply the function to each row and create a new column 'record_metadata'
        evaluations_df["record_metadata"] = evaluations_df.apply(
            extract_metadata, axis=1
        )

        gb = GridOptionsBuilder.from_dataframe(evaluations_df)

        gb.configure_column("app_name", header_name="App Name")
        gb.configure_column("app_version", header_name="App Version")
        gb.configure_column("type", header_name="App Type", hide=True)
        gb.configure_column("record_json", header_name="Record JSON", hide=True)
        gb.configure_column("app_json", header_name="App JSON", hide=True)
        gb.configure_column("cost_json", header_name="Cost JSON", hide=True)
        gb.configure_column("perf_json", header_name="Perf. JSON", hide=True)

        gb.configure_column("record_id", header_name="Record ID", hide=True)
        gb.configure_column("app_id", header_name="App ID", hide=True)

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
            "app_name",
            "app_version",
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
        ] + metadata_cols

        for feedback_col in evaluations_df.columns.drop(non_feedback_cols):
            if "distance" in feedback_col:
                gb.configure_column(
                    feedback_col, hide=feedback_col.endswith("_calls")
                )
            else:
                # cell highlight depending on feedback direction
                cellstyle = JsCode(
                    cellstyle_jscode[
                        feedback_directions.get(feedback_col, default_direction)
                    ]
                )

                gb.configure_column(
                    feedback_col,
                    cellStyle=cellstyle,
                    hide=feedback_col.endswith("_calls"),
                )

        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)

        gridOptions = gb.build()
        data = AgGrid(
            evaluations_df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
        )

        selected_rows = data.selected_rows
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
            record_str = selected_rows["record_json"][0]
            record_json = json.loads(record_str)
            record_metadata = selected_rows["record_metadata"][0]

            app_json = json.loads(
                details
            )  # apps may not be deserializable, don't try to, keep it json.

            row = selected_rows.head().iloc[0]

            st.markdown("#### Feedback results")
            if len(feedback_cols) == 0:
                st.write("No feedback details")
            else:
                feedback_with_valid_results = sorted(
                    list(
                        filter(
                            lambda fcol: row[fcol] is not None, feedback_cols
                        )
                    )
                )

                def get_icon(feedback_name):
                    cat = CATEGORY.of_score(
                        row[feedback_name],
                        higher_is_better=feedback_directions.get(
                            feedback_name, default_direction
                        )
                        == default_direction,
                    )
                    return cat.icon

                icons = list(
                    map(
                        lambda fcol: get_icon(fcol), feedback_with_valid_results
                    )
                )

                selected_fcol = None
                if len(feedback_with_valid_results) > 0:
                    selected_fcol = pills(
                        "Feedback functions (click on a pill to learn more)",
                        feedback_with_valid_results,
                        index=None,
                        format_func=lambda fcol: f"{fcol} {row[fcol]:.4f}",
                        icons=icons,
                    )
                else:
                    st.write("No feedback functions found.")

                def display_feedback_call(call, feedback_name):
                    def highlight(s):
                        if "distance" in feedback_name:
                            return [
                                f"background-color: {CATEGORY.UNKNOWN.color}"
                            ] * len(s)
                        cat = CATEGORY.of_score(
                            s.result,
                            higher_is_better=feedback_directions.get(
                                feedback_name, default_direction
                            )
                            == default_direction,
                        )
                        return [f"background-color: {cat.color}"] * len(s)

                    def highlight_groundedness(s):
                        if "distance" in feedback_name:
                            return [
                                f"background-color: {CATEGORY.UNKNOWN.color}"
                            ] * len(s)
                        cat = CATEGORY.of_score(
                            s.Score,
                            higher_is_better=feedback_directions.get(
                                feedback_name, default_direction
                            )
                            == default_direction,
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
                            args = c["args"]
                            for k, v in args.items():
                                if not isinstance(v, str):
                                    args[k] = pp.pformat(v)

                        df = pd.DataFrame.from_records(c["args"] for c in call)

                        df["result"] = pd.DataFrame([
                            float(call[i]["ret"])
                            if call[i]["ret"] is not None
                            else -1
                            for i in range(len(call))
                        ])
                        df["meta"] = pd.Series([
                            call[i]["meta"] for i in range(len(call))
                        ])
                        df = df.join(
                            df.meta.apply(lambda m: pd.Series(m))
                        ).drop(columns="meta")

                        # note: improve conditional to not rely on the feedback name
                        if "groundedness" in feedback_name.lower():
                            try:
                                # Split the reasons value into separate rows and columns
                                reasons = df["reasons"].iloc[0]
                                # Split the reasons into separate statements
                                statements = reasons.split("STATEMENT")
                                data = []
                                # Each reason has three components: statement, supporting evidence, and score
                                # Parse each reason into these components and add them to the data list
                                for statement in statements[1:]:
                                    try:
                                        criteria = statement.split(
                                            "Criteria: "
                                        )[1].split("Supporting Evidence: ")[0]
                                        supporting_evidence = statement.split(
                                            "Supporting Evidence: "
                                        )[1].split("Score: ")[0]
                                        score_pattern = re.compile(
                                            r"([0-9]+)(?=\D*$)"
                                        )
                                        score_split = statement.split(
                                            "Score: "
                                        )[1]
                                        score_match = score_pattern.search(
                                            score_split
                                        )
                                        if score_match:
                                            score = (
                                                float(score_match.group(1)) / 10
                                            )
                                    except Exception:
                                        pass
                                    data.append({
                                        "Statement": criteria,
                                        "Supporting Evidence from Source": supporting_evidence,
                                        "Score": score,
                                    })
                                reasons_df = pd.DataFrame(data)
                                # Combine the original feedback data with the expanded reasons
                                df_expanded = pd.concat(
                                    [
                                        df.reset_index(drop=True),
                                        reasons_df.reset_index(drop=True),
                                    ],
                                    axis=1,
                                )
                                st.dataframe(
                                    df_expanded.style.apply(
                                        highlight_groundedness, axis=1
                                    ).format("{:.2f}", subset=["Score"]),
                                    hide_index=True,
                                    column_order=[
                                        "Statement",
                                        "Supporting Evidence from Source",
                                        "Score",
                                    ],
                                )
                            except Exception:
                                st.dataframe(
                                    df.style.apply(highlight, axis=1),
                                    hide_index=True,
                                )
                        else:
                            st.dataframe(
                                df.style.apply(highlight, axis=1),
                                hide_index=True,
                            )
                    else:
                        st.text("No feedback details.")

                if selected_fcol is not None:
                    try:
                        if MULTI_CALL_NAME_DELIMITER in selected_fcol:
                            fcol = selected_fcol.split(
                                MULTI_CALL_NAME_DELIMITER
                            )[0]
                        feedback_calls = row[f"{selected_fcol}_calls"]
                        display_feedback_call(feedback_calls, selected_fcol)
                    except Exception:
                        pass

            st.subheader("Trace details")
            val = record_viewer(record_json, app_json)
            st.markdown("")

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
                                0,
                                0.2,
                                0.4,
                                0.6,
                                0.8,
                                1.0,
                            ]  # Quintile buckets
                            ax.hist(
                                app_df[feedback[ind]],
                                bins=bins,
                                edgecolor="black",
                                color="#2D736D",
                            )
                            ax.set_xlabel("Feedback Value")
                            ax.set_ylabel("Frequency")
                            ax.set_title(feedback[ind], loc="center")
                            st.pyplot(fig)
