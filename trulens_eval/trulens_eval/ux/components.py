import streamlit as st
from trulens_eval.tru_db import JSON, get_calls_by_stack

def render_call_frame(frame_json: JSON):
    return f"{'.'.join(frame_json['path'])}.___{frame_json['method_name']}___\n(`{frame_json['module_name']}.{frame_json['class_name']}`)"

def render_calls(record_json: JSON):
    calls = get_calls_by_stack(record_json)
    for call_stack, calls_json in calls.items():
        for call_json in calls_json:

            args = call_json['args']
            rets = call_json['rets']

            frame = call_stack[-1]

            with st.expander(label=render_call_frame(frame)):
            #st.header(render_call_frame(frame))

            # for frame in call_stack[0:-2]:
            #     st.write(render_call_frame(frame))

                st.subheader(f"Inputs:")
                st.write(args)
                st.subheader(f"Outputs:")
                st.write(rets)
