from dotenv import load_dotenv
from langchain_openai import OpenAI
import streamlit as st
from trulens.core import Feedback
from trulens.core import Tru
import trulens.dashboard.streamlit as trulens_st
from trulens.ext.instrument.langchain import TruChain
from trulens.ext.provider.openai import OpenAI as fOpenAI

load_dotenv()

tru = Tru()

st.title("🦑 Using TruLens Components in Streamlit")

provider = fOpenAI()

f_coherence = Feedback(provider.coherence_with_cot_reasons).on_output()

feedbacks = [f_coherence]


def generate_response(input_text):
    llm = OpenAI(temperature=0.7)
    tru_llm = TruChain(llm, app_id="LLM v1", feedbacks=feedbacks)
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
