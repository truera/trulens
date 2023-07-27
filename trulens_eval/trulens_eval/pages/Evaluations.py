import json
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from streamlit_javascript import st_javascript
from ux.add_logo import add_logo
from ux.styles import CATEGORY

from trulens_eval import Tru
from trulens_eval.app import ComponentView
from trulens_eval.app import instrumented_component_views
from trulens_eval.app import LLM
from trulens_eval.app import Other
from trulens_eval.app import Prompt
from trulens_eval.react_components.record_viewer import record_viewer
from trulens_eval.schema import Record
from trulens_eval.schema import Select
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.ux.components import draw_call
from trulens_eval.ux.components import draw_llm_info
from trulens_eval.ux.components import draw_metadata
from trulens_eval.ux.components import draw_prompt_info
from trulens_eval.ux.components import render_selector_markdown
from trulens_eval.ux.components import write_or_json
from trulens_eval.ux.styles import cellstyle_jscode

st.set_page_config(page_title="Evaluations", layout="wide")

st.title("Evaluations")

st.runtime.legacy_caching.clear_cache()

add_logo()

tru = Tru()
lms = tru.db

df_results, feedback_cols = lms.get_records_and_feedback([])

state = st.session_state

if "clipboard" not in state:
    state.clipboard = "nothing"

if state.clipboard:
    ret = st_javascript(
        f"""navigator.clipboard.writeText("{state.clipboard}")
    .then(
        function() {{
            console.log('success?')
        }},
        function(err) {{
            console.error("Async: Could not copy text: ", err)
        }}
    )
"""
    )


def jsonify_for_ui(*args, **kwargs):
    return jsonify(*args, **kwargs, redact_keys=True, skip_specials=True)


def render_component(query, component, header=True):
    # Draw the accessor/path within the wrapped app of the component.
    if header:
        st.subheader(
            f"Component {render_selector_markdown(Select.for_app(query))}"
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

    elif isinstance(component, Other):
        with st.expander("Uncategorized Component Details:"):
            st.json(jsonify_for_ui(component.json))

    else:
        with st.expander("Unhandled Component Details:"):
            st.json(jsonify_for_ui(component.json))


if df_results.empty:
    st.write("No records yet...")

else:
    apps = list(df_results.app_id.unique())
    if 'app' in st.session_state:
        app = st.session_state.app
    else:
        app = apps

    options = st.multiselect('Filter Applications', apps, default=app)

    if (len(options) == 0):
        st.header("All Applications")
        app_df = df_results

    elif (len(options) == 1):
        st.header(options[0])

        app_df = df_results[df_results.app_id.isin(options)]

    else:
        st.header("Multiple Applications Selected")

        app_df = df_results[df_results.app_id.isin(options)]

    tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

    with tab1:

        gridOptions = {'alwaysShowHorizontalScroll': True}
        evaluations_df = app_df
        gb = GridOptionsBuilder.from_dataframe(evaluations_df)

        cellstyle_jscode = JsCode(cellstyle_jscode)
        gb.configure_column('type', header_name='App Type')
        gb.configure_column('record_json', header_name='Record JSON', hide=True)
        gb.configure_column('app_json', header_name='App JSON', hide=True)
        gb.configure_column('cost_json', header_name='Cost JSON', hide=True)
        gb.configure_column('perf_json', header_name='Perf. JSON', hide=True)

        gb.configure_column('record_id', header_name='Record ID', hide=True)
        gb.configure_column('app_id', header_name='App ID')

        gb.configure_column('feedback_id', header_name='Feedback ID', hide=True)
        gb.configure_column('input', header_name='User Input')
        gb.configure_column(
            'output',
            header_name='Response',
        )
        gb.configure_column('total_tokens', header_name='Total Tokens (#)')
        gb.configure_column('total_cost', header_name='Total Cost (USD)')
        gb.configure_column('latency', header_name='Latency (Seconds)')
        gb.configure_column('tags', header_name='Tags')
        gb.configure_column('ts', header_name='Time Stamp', sort="desc")

        non_feedback_cols = [
            'app_id', 'type', 'ts', 'total_tokens', 'total_cost', 'record_json',
            'latency', 'record_id', 'app_id', 'cost_json', 'app_json', 'input',
            'output', 'perf_json'
        ]

        for feedback_col in evaluations_df.columns.drop(non_feedback_cols):
            gb.configure_column(
                feedback_col,
                cellStyle=cellstyle_jscode,
                hide=feedback_col.endswith("_calls")
            )
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        #gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        data = AgGrid(
            evaluations_df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True
        )

        selected_rows = data['selected_rows']
        selected_rows = pd.DataFrame(selected_rows)

        if len(selected_rows) == 0:
            st.write("Hint: select a row to display details of a record")

        else:
            st.header(f"Selected LLM Application: {selected_rows['app_id'][0]}")
            st.text(f"Selected Record ID: {selected_rows['record_id'][0]}")

            prompt = selected_rows['input'][0]
            response = selected_rows['output'][0]
            details = selected_rows['app_json'][0]

            app_json = json.loads(
                details
            )  # apps may not be deserializable, don't try to, keep it json.
            with st.expander(
                    f"Input {render_selector_markdown(Select.RecordInput)}",
                    expanded=True):
                write_or_json(st, obj=prompt)

            with st.expander(
                    f"Response {render_selector_markdown(Select.RecordOutput)}",
                    expanded=True):
                write_or_json(st, obj=response)

            metadata = app_json.get('metadata')
            if metadata:
                with st.expander("Metadata"):
                    st.markdown(draw_metadata(metadata))

            row = selected_rows.head().iloc[0]

            st.header("Feedback")
            for fcol in feedback_cols:
                feedback_name = fcol
                feedback_result = row[fcol]
                feedback_calls = row[f"{fcol}_calls"]

                def display_feedback_call(call):

                    def highlight(s):
                        cat = CATEGORY.of_score(s.result)
                        return [f'background-color: {cat.color}'] * len(s)

                    if call is not None and len(call) > 0:

                        df = pd.DataFrame.from_records(
                            [call[i]["args"] for i in range(len(call))]
                        )
                        df["result"] = pd.DataFrame(
                            [
                                float(call[i]["ret"] or -1)
                                for i in range(len(call))
                            ]
                        )
                        df["meta"] = pd.Series(
                            [call[i]["meta"] for i in range(len(call))]
                        )
                        df = df.join(df.meta.apply(lambda m: pd.Series(m))
                                    ).drop(columns="meta")

                        st.dataframe(
                            df.style.apply(highlight, axis=1
                                          ).format("{:.2}", subset=["result"])
                        )

                    else:
                        st.text("No feedback details.")

                with st.expander(f"{feedback_name} = {feedback_result}",
                                 expanded=True):
                    display_feedback_call(feedback_calls)

            record_str = selected_rows['record_json'][0]
            record_json = json.loads(record_str)
            record = Record(**record_json)

            classes: Iterable[Tuple[JSONPath, ComponentView]
                             ] = list(instrumented_component_views(app_json))
            classes_map = {path: view for path, view in classes}

            st.header('Timeline')
            val = record_viewer(record_json, app_json)

            match_query = None

            # Assumes record_json['perf']['start_time'] is always present
            if val != record_json['perf']['start_time'] and val != '':
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
                            f"Call by {match_query} was not associated with any instrumented component."
                        )
                        # Look up whether there was any data at that path even if not an instrumented component:
                        app_component_json = list(match_query(app_json))[0]
                        if app_component_json is not None:
                            with st.expander(
                                    "Uninstrumented app component details."):
                                st.json(app_component_json)

                else:
                    st.text('No match found')
            else:
                st.subheader(f"App {render_selector_markdown(Select.App)}")
                with st.expander("App Details:"):
                    st.json(jsonify_for_ui(app_json))

            if match_query is not None:
                st.header("Subcomponents:")

                for query, component in classes:
                    if not match_query.is_immediate_prefix_of(query):
                        continue

                    if len(query.path) == 0:
                        # Skip App, will still list App.app under "app".
                        continue

                    render_component(query, component)

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
                                edgecolor='black',
                                color='#2D736D'
                            )
                            ax.set_xlabel('Feedback Value')
                            ax.set_ylabel('Frequency')
                            ax.set_title(feedback[ind], loc='center')
                            st.pyplot(fig)
