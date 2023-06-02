from typing import Dict

import streamlit as st

from trulens_eval.tru_db import get_calls_by_stack
from trulens_eval.tru_db import JSON


def render_call_frame(frame_json: JSON):
    return f"{'.'.join(frame_json['path'])}.___{frame_json['method_name']}___\n(`{frame_json['module_name']}.{frame_json['class_name']}`)"


def render_calls(record_json: JSON, index: int):
    calls = get_calls_by_stack(record_json)

    chain_step = 0
    for call_stack, calls_json in calls.items():
        chain_step += 1
        if chain_step == index:
            for call_json in calls_json:
                args = call_json['args']
                rets = call_json['rets']

                frame = call_stack[-1]

                with st.expander(label=render_call_frame(frame)):
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
