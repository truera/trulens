from dotenv import load_dotenv
from langchain_openai import OpenAI
import streamlit as st
from trulens.apps.langchain import tru_chain as mod_tru_chain
from trulens.core import session as core_session
from trulens.core.feedback import feedback as core_feedback
import trulens.dashboard.streamlit as trulens_st
from trulens.providers.openai import provider as openai_provider

load_dotenv()

session = core_session.TruSession()

st.title("🦑 Using TruLens Components in Streamlit")

provider = openai_provider.OpenAI()

f_coherence = core_feedback.Feedback(
    provider.coherence_with_cot_reasons
).on_output()

feedbacks = [f_coherence]


def generate_response(input_text):
    llm = OpenAI(temperature=0.7)
    tru_llm = mod_tru_chain.TruChain(
        llm, app_name="LLM v1", feedbacks=feedbacks
    )
    with tru_llm as recording:
        response = llm.invoke(input_text)
    record = recording.get()
    return record, response


with st.form("my_form"):
    text = st.text_area(
        "Enter text:", "What are 3 key advice for learning how to code?"
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        record, response = generate_response(text)
        st.info(response)

if submitted:
    trulens_st.trulens_feedback(record=record)
    trulens_st.trulens_trace(record=record)
