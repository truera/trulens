import streamlit as st

import streamlit as st
from langchain_openai import OpenAI

from trulens_eval import TruChain
from trulens_eval import Tru
from trulens_eval import Feedback
from trulens_eval.feedback.provider import OpenAI as fOpenAI
from trulens_eval.streamlit import feedback as st_feedback

import os
from dotenv import load_dotenv

load_dotenv()

tru = Tru()

st.title("🦑 Using TruLens Components in Streamlit")

provider = fOpenAI()

f_coherence = Feedback(provider.coherence_with_cot_reasons).on_output()

feedbacks = [f_coherence]

def generate_response(input_text):
    llm = OpenAI(temperature=0.7)
    tru_llm = TruChain(llm, app_id = "LLM v1", feedbacks=feedbacks)
    with tru_llm as recording:
        response = llm.invoke(input_text)
    record = recording.get()
    return record, response

with st.form("my_form"):
    text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        record, response = generate_response(text)
        st.info(response)

if submitted:
    st_feedback.trulens_feedback(record = record)
