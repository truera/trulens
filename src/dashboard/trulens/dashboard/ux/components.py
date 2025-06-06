import json
import random
from typing import Dict, List

import pandas as pd
import streamlit as st
from trulens.core import app as core_app
from trulens.core.schema import record as record_schema
from trulens.core.schema import select as select_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import constants as constant_utils
from trulens.core.utils import containers as container_utils
from trulens.core.utils import json as json_utils
from trulens.core.utils import keys as key_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.dashboard import constants as dashboard_constants


def write_or_json(st, obj):
    """
    Dispatch either st.json or st.write depending on content of `obj`. If it is
    a string that can parses into strictly json (dict), use st.json, otherwise
    use st.write.
    """

    if isinstance(obj, str):
        try:
            content = json.loads(obj)
            if not isinstance(content, str):
                st.json(content)
            else:
                st.write(content)

        except BaseException:
            st.write(obj)


def copy_to_clipboard(path, *args, **kwargs):
    st.session_state.clipboard = str(path)


def draw_selector_button(path) -> None:
    st.button(
        key=str(random.random()),
        label=f"{select_schema.Select.render_for_dashboard(path)}",
        on_click=copy_to_clipboard,
        args=(path,),
    )


def render_selector_markdown(path) -> str:
    return f"[`{select_schema.Select.render_for_dashboard(path)}`]"


def render_call_frame(
    frame: record_schema.RecordAppCall, path=None
) -> str:  # markdown
    path = path or frame.path

    return f"__{frame.method.name}__ (__{frame.method.obj.cls.module.module_name}.{frame.method.obj.cls.name}__)"


def dict_to_md(dictionary: dict) -> str:
    if len(dictionary) == 0:
        return "No metadata."
    mdheader = "|"
    mdseparator = "|"
    mdbody = "|"
    for key, value in dictionary.items():
        mdheader = mdheader + str(key) + "|"
        mdseparator = mdseparator + "-------|"
        mdbody = mdbody + str(value) + "|"
    mdtext = mdheader + "\n" + mdseparator + "\n" + mdbody
    return mdtext


def draw_metadata_and_tags(
    metadata: types_schema.Metadata, tags: types_schema.Tags
) -> str:
    if isinstance(metadata, Dict):
        metadata["tags"] = tags

        # Make the boolean column names more user-friendly
        if metadata.get(dashboard_constants.EXTERNAL_APP_COL_NAME) is not None:
            metadata["External App?"] = metadata.pop(
                dashboard_constants.EXTERNAL_APP_COL_NAME
            )

        if metadata.get(dashboard_constants.PINNED_COL_NAME) is not None:
            metadata["Pinned?"] = metadata.pop(
                dashboard_constants.PINNED_COL_NAME
            )

        return dict_to_md(metadata)
    else:
        # We shouldn't ever get here since types_schema.Metadata is a Dict.
        return str(metadata) + "\n" + str(tags)


def draw_call(call: record_schema.RecordAppCall) -> None:
    top = call.stack[-1]

    path = select_schema.Select.for_record(
        top.path._append(
            step=serial_utils.GetItemOrAttribute(
                item_or_attribute=top.method.name
            )
        )
    )

    with st.expander(
        label="Call "
        + render_call_frame(top, path=path)
        + " "
        + render_selector_markdown(path)
    ):
        args = call.args
        rets = call.rets

        for frame in call.stack[::-1][1:]:
            st.write("Via " + render_call_frame(frame, path=path))

        st.subheader(f"Inputs {render_selector_markdown(path.args)}")
        if isinstance(args, Dict):
            st.json(args)
        else:
            st.write(args)

        st.subheader(f"Outputs {render_selector_markdown(path.rets)}")
        if isinstance(rets, Dict):
            st.json(rets)
        else:
            st.write(rets)


def draw_calls(record: record_schema.Record, index: int) -> None:
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


def draw_prompt_info(
    query: serial_utils.Lens, component: core_app.ComponentView
) -> None:
    prompt_details_json = json_utils.jsonify(component.json, skip_specials=True)

    st.caption("Prompt details")

    path = select_schema.Select.for_app(query)

    prompt_types = {
        k: v
        for k, v in prompt_details_json.items()
        if (v is not None)
        and not container_utils.is_empty(v)
        and not pyschema_utils.is_noserio(v)
        and k != constant_utils.CLASS_INFO
    }

    for key, value in prompt_types.items():
        with st.expander(
            key.capitalize()
            + " "
            + render_selector_markdown(getattr(path, key)),
            expanded=True,
        ):
            if isinstance(value, (Dict, List)):
                st.write(value)
            else:
                if isinstance(value, str) and len(value) > 32:
                    st.text(value)
                else:
                    st.write(value)


def draw_llm_info(
    query: serial_utils.Lens, component: core_app.ComponentView
) -> None:
    llm_details_json = component.json

    st.subheader("*LLM Details*")
    # path_str = str(query)
    # st.text(path_str[:-4])

    llm_kv = {
        k: v
        for k, v in llm_details_json.items()
        if (v is not None)
        and not container_utils.is_empty(v)
        and not pyschema_utils.is_noserio(v)
        and k != constant_utils.CLASS_INFO
    }
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    df = pd.DataFrame.from_dict(llm_kv, orient="index").transpose()

    # Redact any column whose name indicates it might be a secret.
    for col in df.columns:
        if key_utils.should_redact_key(col):
            df[col] = key_utils.REDACTED_VALUE

    # TODO: What about columns not indicating a secret but some values do
    # indicate it as per `should_redact_value` ?

    # Iterate over each column of the DataFrame
    for column in df.columns:
        path = getattr(select_schema.Select.for_app(query), str(column))
        # Check if any cell in the column is a dictionary

        if any(isinstance(cell, dict) for cell in df[column]):
            # Create new columns for each key in the dictionary
            new_columns = df[column].apply(
                lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series()
            )
            new_columns.columns = [
                f"{key} {render_selector_markdown(path)}"
                for key in new_columns.columns
            ]

            # Remove extra zeros after the decimal point
            new_columns = new_columns.applymap(
                lambda x: "{0:g}".format(x) if isinstance(x, float) else x
            )

            # Add the new columns to the original DataFrame
            df = pd.concat([df.drop(column, axis=1), new_columns], axis=1)

        else:
            # TODO: add selectors to the output here

            pass

    # Inject CSS with Markdown

    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df)


def draw_agent_info(
    query: serial_utils.Lens, component: core_app.ComponentView
) -> None:
    # copy of draw_prompt_info
    # TODO: dedup
    prompt_details_json = json_utils.jsonify(component.json, skip_specials=True)

    st.subheader("*Agent Details*")

    path = select_schema.Select.for_app(query)

    prompt_types = {
        k: v
        for k, v in prompt_details_json.items()
        if (v is not None)
        and not container_utils.is_empty(v)
        and not pyschema_utils.is_noserio(v)
        and k != constant_utils.CLASS_INFO
    }

    for key, value in prompt_types.items():
        with st.expander(
            key.capitalize()
            + " "
            + render_selector_markdown(getattr(path, key)),
            expanded=True,
        ):
            if isinstance(value, (Dict, List)):
                st.write(value)
            else:
                if isinstance(value, str) and len(value) > 32:
                    st.text(value)
                else:
                    st.write(value)


def draw_tool_info(
    query: serial_utils.Lens, component: core_app.ComponentView
) -> None:
    # copy of draw_prompt_info
    # TODO: dedup
    prompt_details_json = json_utils.jsonify(component.json, skip_specials=True)

    st.subheader("*Tool Details*")

    path = select_schema.Select.for_app(query)

    prompt_types = {
        k: v
        for k, v in prompt_details_json.items()
        if (v is not None)
        and not container_utils.is_empty(v)
        and not pyschema_utils.is_noserio(v)
        and k != constant_utils.CLASS_INFO
    }

    for key, value in prompt_types.items():
        with st.expander(
            key.capitalize()
            + " "
            + render_selector_markdown(getattr(path, key)),
            expanded=True,
        ):
            if isinstance(value, (Dict, List)):
                st.write(value)
            else:
                if isinstance(value, str) and len(value) > 32:
                    st.text(value)
                else:
                    st.write(value)
