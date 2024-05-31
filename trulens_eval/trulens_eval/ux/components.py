import json
import random
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from trulens_eval.app import ComponentView
from trulens_eval.keys import REDACTED_VALUE
from trulens_eval.keys import should_redact_key
from trulens_eval.schema.feedback import Select
from trulens_eval.schema.record import Record
from trulens_eval.schema.record import RecordAppCall
from trulens_eval.schema.types import Metadata
from trulens_eval.utils.containers import is_empty
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import CLASS_INFO
from trulens_eval.utils.pyschema import is_noserio
from trulens_eval.utils.serial import GetItemOrAttribute
from trulens_eval.utils.serial import JSON_BASES
from trulens_eval.utils.serial import Lens


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
        label=f"{Select.render_for_dashboard(path)}",
        on_click=copy_to_clipboard,
        args=(path,)
    )


def render_selector_markdown(path) -> str:
    return f"[`{Select.render_for_dashboard(path)}`]"


def render_call_frame(frame: RecordAppCall, path=None) -> str:  # markdown
    path = path or frame.path

    return (
        f"__{frame.method.name}__ (__{frame.method.obj.cls.module.module_name}.{frame.method.obj.cls.name}__)"
    )


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


def draw_metadata(metadata: Metadata) -> str:
    if isinstance(metadata, Dict):
        return dict_to_md(metadata)
    else:
        return str(metadata)


def draw_call(call: RecordAppCall) -> None:
    top = call.stack[-1]

    path = Select.for_record(
        top.path._append(
            step=GetItemOrAttribute(item_or_attribute=top.method.name)
        )
    )

    with st.expander(label=f"Call " + render_call_frame(top, path=path) + " " +
                     render_selector_markdown(path)):

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