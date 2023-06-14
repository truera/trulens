# Run with command: streamlit run Example_Thumbs_Application.py
import langchain
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
import streamlit as st

import sys
import os
from pathlib import Path

dev_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, dev_path)
os.environ["OPENAI_API_KEY"] = "..."

from trulens_eval import TruChain, Tru
# Set up GPT-3 model
model_name = "gpt-3.5-turbo"
tru = Tru()


# Define function to generate GPT-3 response
@st.cache_resource
def setup_chain():
    full_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=
            "Provide a helpful response with relevant background information for the following: {prompt}",
            input_variables=["prompt"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

    chat = ChatOpenAI(model_name=model_name, temperature=0.9)

    chain = LLMChain(llm=chat, prompt=chat_prompt_template)

    tc = TruChain(chain, chain_id='Streamlit App')
    tru.add_chain(chain=tc)
    tru.run_dashboard(_dev=dev_path)
    return tc


def generate_response(prompt, tc):
    return tc.call_with_record(prompt)


tc = setup_chain()

# Set up Streamlit app
st.title("Get Help from ChatGPT")
user_input = st.text_input("What do you need help with?")

if user_input:
    # Generate GPT-3 response
    prompt_input = user_input
    # add context manager to capture tokens and cost of the chain
    gpt3_response, record = generate_response(prompt_input, tc)

    # Display response
    st.write("Here's some help for you:")
    st.write(gpt3_response['text'])

    # Allow user to rate the response with emojis
    col1, col2 = st.columns(2)
    with col1:
        thumbs_up = st.button("👍")
    with col2:
        thumbs_down = st.button("👎")

    thumb_result = None
    if thumbs_up:
        st.write("Thank you for your feedback! We're glad we could help.")
        thumb_result = True
    elif thumbs_down:
        # Save rating to database or file
        st.write(
            "We're sorry we couldn't be more helpful. Please try again with a different question."
        )
        thumb_result = False
    if thumb_result is not None:
        tru.add_feedback(
            name="👍 (1) or 👎 (0)",
            record_id=record.record_id,
            chain_id=tc.chain_id,
            result=thumb_result
        )
