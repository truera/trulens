from typing import Dict, List

import pandas as pd
import streamlit as st

from trulens_eval.schema import Record
from trulens_eval.schema import RecordAppCall
from trulens_eval.db import JSON
from trulens_eval.app import ComponentView
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.util import CLASS_INFO
from trulens_eval.util import is_empty
from trulens_eval.util import is_noserio


def render_call_frame(frame: RecordAppCall) -> str:  # markdown

    return (
        f"{frame.path}.___{frame.method.name}___\n"
        f"(`{frame.method.obj.cls.module.module_name}.{frame.method.obj.cls.name}`)"
    )


def draw_call(call) -> None:
    top = call.stack[-1]

    with st.expander(label=render_call_frame(top)):
        args = call.args
        rets = call.rets

        for frame in call.stack[0:-2]:
            st.write("Via " + render_call_frame(frame))

        st.subheader(f"Inputs:")
        if isinstance(args, Dict):
            st.json(args)
        else:
            st.write(args)

        st.subheader(f"Outputs:")
        if isinstance(rets, Dict):
            st.json(rets)
        else:
            st.write(rets)


def draw_calls(record: Record, index: int) -> None:
    """
    Draw the calls recorded in a `record`.
    """

    calls = record.calls

    app_step = 0

    for call in calls:
        app_step += 1

        if app_step != index:
            continue

        draw_call(call)


def draw_prompt_info(query: JSONPath, component: ComponentView) -> None:
    prompt_details_json = jsonify(component.json, skip_specials=True)

    path_str = str(query)
    st.subheader(f"*Prompt Details*")

    prompt_types = {
        k: v for k, v in prompt_details_json.items() if (v is not None) and
        not is_empty(v) and not is_noserio(v) and k != CLASS_INFO
    }

    for key, value in prompt_types.items():
        with st.expander(key.capitalize(), expanded=True):
            if isinstance(value, (Dict, List)):
                st.write(value)
            else:
                if isinstance(value, str) and len(value) > 32:
                    st.text(value)
                else:
                    st.write(value)


def draw_llm_info(query: JSONPath, component: ComponentView) -> None:
    llm_details_json = component.json

    st.subheader(f"*LLM Details*")
    # path_str = str(query)
    # st.text(path_str[:-4])

    llm_kv = {
        k: v for k, v in llm_details_json.items() if (v is not None) and
        not is_empty(v) and not is_noserio(v) and k != CLASS_INFO
    }
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    df = pd.DataFrame.from_dict(llm_kv, orient='index').transpose()

    # Iterate over each column of the DataFrame
    for column in df.columns:
        # Check if any cell in the column is a dictionary
        if any(isinstance(cell, dict) for cell in df[column]):
            # Create new columns for each key in the dictionary
            new_columns = df[column].apply(
                lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series()
            )
            new_columns.columns = [f"{key}" for key in new_columns.columns]

            # Remove extra zeros after the decimal point
            new_columns = new_columns.applymap(
                lambda x: '{0:g}'.format(x) if isinstance(x, float) else x
            )

            # Add the new columns to the original DataFrame
            df = pd.concat([df.drop(column, axis=1), new_columns], axis=1)
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df)
