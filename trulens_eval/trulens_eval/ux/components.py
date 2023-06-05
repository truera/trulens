from typing import Dict

import streamlit as st

from trulens_eval.schema import Record
from trulens_eval.schema import RecordChainCall
from trulens_eval.tru_db import get_calls_by_stack
from trulens_eval.tru_db import JSON


def render_call_frame(frame: RecordChainCall) -> str:  # markdown

    return (
        f"{frame.path}.___{frame.method.method_name}___\n"
        f"(`{frame.method.module_name}.{frame.method.class_name}`)"
    )


def draw_calls(record: Record, index: int) -> None:
    """
    Draw the calls recorded in a `record`.
    """

    calls = record.calls

    chain_step = 0

    for call in calls:
        chain_step += 1
        top = call.chain_stack[-1]

        if chain_step != index:
            continue

        with st.expander(label=render_call_frame(top)):
            args = call.args
            rets = call.rets

            for frame in call.chain_stack[0:-2]:
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
