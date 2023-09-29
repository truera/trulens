import asyncio
from typing import Optional

import pydantic

from trulens_eval.utils.serial import JSON
from trulens_eval.ux.apps import ChatRecord

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())
import streamlit as st
from ux.add_logo import add_logo

st.set_page_config(page_title="App Runner", layout="wide")

st.runtime.legacy_caching.clear_cache()

add_logo()

from trulens_eval.schema import AppDefinition

# state = st.session_state
if "records" not in st.session_state:
    st.title("App Runner")

    for app_json in AppDefinition.get_loadable_apps():
        st.write(app_json['app_id'])
        select_app = st.button("New Session")

        if select_app:
            st.write("Loading ...")
            st.session_state.records = [ChatRecord(app_json=app_json)]
            st.rerun()
    
else:
    first_record = st.session_state.records[0]
    app_json = first_record.app_json

    st.title(f"App Runner: {app_json['app_id']}")
    st.write(f"TODO: link to {app_json['app_id']} on other pages.")

    for rec in st.session_state.records:
        col1, col2 = st.columns(2)

        if isinstance(rec, dict):
            rec = ChatRecord(**rec)
        
        record_json = rec.record_json

        # assert isinstance(rec, ChatRecord), f"rec={type(rec).__name__}"
        with col1:
            if rec.human is not None:
                with st.chat_message("Human", avatar="🧑‍💻"):
                    st.write(rec.human)
                    # st.write(f"TODO link to {rec.record_json['record_id']} focusing on `main_input`.")
            if rec.computer is not None:
                with st.chat_message("Computer", avatar="🤖"):
                    st.write(rec.computer)
                    # st.write(f"TODO link to {rec.record_json['record_id']} focusing on `main_output`.")
        with col2:
            if record_json is not None:
                st.write(f"TODO link to {record_json['record_id']}.")

    human_input = st.chat_input()

    if human_input:
        col1, col2 = st.columns(2)

        with col1:
            with st.chat_message("Human", avatar="🧑‍💻"):
                st.write(human_input)

            current_record = st.session_state.records[-1]

            tru_app = AppDefinition.continue_session(
                app_definition_json=current_record.app_json
            )

            with tru_app as rec:
                comp_response = tru_app.main_call(human_input)
            record = rec.get()

            with st.chat_message("Computer", avatar="🤖"):
                st.write(comp_response)

        with col2:
            st.write(f"TODO link to {record.record_id}.")

        # Update current ChatRecord with the results.
        current_record.human = human_input
        current_record.record_json = record.dict()
        current_record.computer = comp_response

        # Add the next ChatRecord that contains the updated app state:
        st.session_state.records.append(ChatRecord(app_json=tru_app.dict()))
