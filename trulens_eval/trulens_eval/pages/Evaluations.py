import json
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from trulens_eval.app import ComponentView
from trulens_eval.app import LLM, Memory, Other, Prompt
from trulens_eval.app import instrumented_component_views
from trulens_eval.util import jsonify
from ux.add_logo import add_logo
from ux.styles import default_pass_fail_color_threshold

from trulens_eval import Tru
from trulens_eval.schema import Record
from trulens_eval.util import JSONPath
from trulens_eval.ux.components import draw_call
from trulens_eval.ux.components import draw_llm_info
from trulens_eval.ux.components import draw_prompt_info
from trulens_eval.ux.styles import cellstyle_jscode

st.set_page_config(page_title="Evaluations", layout="wide")

st.title("Evaluations")

st.runtime.legacy_caching.clear_cache()

add_logo()

tru = Tru()
lms = tru.db

df_results, feedback_cols = lms.get_records_and_feedback([])

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
            st.write("Hint: select a row to display app metadata")

        else:
            st.header(f"Selected LLM Application: {selected_rows['app_id'][0]}")
            st.text(f"Selected Record ID: {selected_rows['record_id'][0]}")

            prompt = selected_rows['input'][0]
            response = selected_rows['output'][0]

            with st.expander("Input Prompt", expanded=True):
                st.write(prompt)

            with st.expander("Response", expanded=True):
                st.write(response)

            row = selected_rows.head().iloc[0]

            st.header("Feedback")
            for fcol in feedback_cols:
                feedback_name = fcol
                feedback_result = row[fcol]
                feedback_calls = row[f"{fcol}_calls"]

                def display_feedback_call(call):

                    def highlight(s):
                        return ['background-color: #4CAF50'] * len(
                            s
                        ) if s.result >= default_pass_fail_color_threshold else [
                            'background-color: #FCE6E6'
                        ] * len(s)

                    if call is not None and len(call) > 0:
                        df = pd.DataFrame.from_records(
                            [call[i]["args"] for i in range(len(call))]
                        )
                        df["result"] = pd.DataFrame(
                            [float(call[i]["ret"]) for i in range(len(call))]
                        )
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

            details = selected_rows['app_json'][0]
            app_json = json.loads(
                details
            )  # apps may not be deserializable, don't try to, keep it json.

            classes: Iterable[Tuple[JSONPath, List[ComponentView]]
                             ] = instrumented_component_views(app_json)

            st.header("Components")

            for query, component in classes:

                if len(query.path) == 0:
                    # Skip App, will still list A.app under "app" below.
                    continue

                # Draw the accessor/path within the wrapped app of the component.
                st.subheader(f"{query}")

                # Draw the python class information of this component.
                cls = component.cls
                base_cls = cls.base_class()
                label = f"`{repr(cls)}`"
                if str(base_cls) != str(cls):
                    label += f" < `{repr(base_cls)}`"
                st.write(label)

                # Per-component-type drawing routines.
                if isinstance(component, LLM):
                    draw_llm_info(component=component, query=query)

                elif isinstance(component, Prompt):
                    draw_prompt_info(component=component, query=query)

                elif isinstance(component, Other):
                    with st.expander("Uncategorized Component Details:"):
                        st.json(jsonify(component.json, skip_specials=True))

                else:
                    with st.expander("Unhandled Component Details:"):
                        st.json(jsonify(component.json, skip_specials=True))

                # Draw the calls issued to component.
                calls = [
                    call for call in record.calls
                    if query == call.stack[-1].path
                ]
                if len(calls) > 0:
                    st.subheader("Calls to component:")
                    for call in calls:
                        draw_call(call)

            st.header("More options:")

            if st.button("Display full app json"):

                st.write(jsonify(app_json, skip_specials=True))

            if st.button("Display full record json"):

                st.write(jsonify(record_json, skip_specials=True))

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
