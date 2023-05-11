import dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
#mport openai
import streamlit as st

from keys import *
import tru
import tru_feedback
import tru_chain

# Set up GPT-3 model
model_engine = "gpt-3.5-turbo"


# Define function to generate GPT-3 response
@st.cache_data
def generate_response(prompt, model_name):
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
    tc = tru_chain.TruChain(chain)
    return tc(prompt)


# Set up Streamlit app
st.title("Get Help from ChatGPT")
user_input = st.text_input("What do you need help with?")

if user_input:
    # Generate GPT-3 response
    prompt_input = user_input
    # add context manager to capture tokens and cost of the chain
    with get_openai_callback() as cb:
        gpt3_response, record = generate_response(prompt_input, model_engine)
        total_tokens = cb.total_tokens
        total_cost = cb.total_cost

    # Display response
    st.write("Here's some help for you:")
    st.write(gpt3_response)

    # Allow user to rate the response with emojis
    col1, col2 = st.columns(2)
    with col1:
        thumbs_up = st.button("👍")
    with col2:
        thumbs_down = st.button("👎")

    if thumbs_up:
        # Save rating to database or file
        st.write("Thank you for your feedback! We're glad we could help.")
    elif thumbs_down:
        # Save rating to database or file
        st.write(
            "We're sorry we couldn't be more helpful. Please try again with a different question."
        )

    record_id = tru.add_data(
        'chat_model',
        prompt_input,
        gpt3_response,
        record,
        '',
        total_tokens=total_tokens,
        total_cost=total_cost
    )

    # Run feedback function and get value
    feedback = tru.run_feedback_function(
        prompt_input, gpt3_response, [
            tru_feedback.get_hate_function(
                evaluation_choice='response',
                provider='openai',
                model_engine='moderation'
            )
        ]
    )

    # Add value to database
    tru.add_feedback(record_id, feedback)
