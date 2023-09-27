import asyncio

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
    print("resetting state")
    st.session_state.records = []

if "i" not in st.session_state:
    st.session_state['i'] = 0

if 'app_args' not in st.session_state:

    st.title("App Runner")

    for app_args in AppDefinition.get_loadable_apps():

        app_json = app_args['app_definition_json']
        st.write(app_json['app_id'])
        select_app = st.button("New Session")
        if select_app:
            st.write("selected")
            st.session_state['app_args'] = app_args
            st.session_state.records = []
            st.rerun()


else:
    app_args = st.session_state['app_args']
    tru_app = AppDefinition.new_session(**app_args)

    st.title(f"App Runner: {tru_app.app_id}")

    human_key = f"human_input"
    human_input = st.text_input(key=human_key, label=f"human")#, disabled=st.session_state[human_key + "_disabled"])

    for record in st.session_state.records:
        h, c = record
        if h is not None:
            st.text(f"Human: {h}")
        if c is not None:
            st.text(f"Computer: {c}")
        

    st.session_state.records.append([None, None])

    #def new_record():
    # i = st.session_state["i"]
    # print(len(st.session_state.records))
    #st.session_state[human_key + "_disabled"] = False
    #    human_input = st.text_input(key=human_key, label=f"human {i}", disabled=st.session_state[human_key + "_disabled"])

    if human_input:
        st.session_state.records[-1][0] = human_input
        st.text(f"Human: {human_input}")
        # st.session_state[human_key + "_disabled"] = True
        # st.session_state.records[i][0] = True
        # st.session_state.records[i][1] = human_input

        with tru_app as rec:
            comp_response = tru_app.main_call(human_input)
            # st.session_state.records[i][2] = comp_response
            #st.text_input(key=f"computer_input_{i}", label=f"computer {i}", value=comp_response, disabled=True)
            # st.text(f"computer: {comp_response}")

        # st.session_state['i'] += 1
        st.text(f"Computer: {comp_response}")
        st.session_state.records[-1][1] = comp_response
        st.session_state.records.append([None, None])
        # print("pre", len(st.session_state.records))
        # st.session_state.records.append([False, None, None])
        # print("post", len(st.session_state.records))

        #new_record()

    #new_record()