import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import Pinecone
import pinecone
import streamlit as st

from trulens_evalchain import tru
from trulens_evalchain import tru_chain
from trulens_evalchain.keys import *
from trulens_evalchain.keys import PINECONE_API_KEY
from trulens_evalchain.keys import PINECONE_ENV

# Set up GPT-3 model
model_name = "gpt-3.5-turbo"

chain_id = "TruBot_langprompt"

# Pinecone configuration.
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)

identity = lambda h: h


# @st.cache_data
def generate_response(prompt):
    # Embedding needed for Pinecone vector db.
    embedding = OpenAIEmbeddings(model='text-embedding-ada-002')  # 1536 dims
    docsearch = Pinecone.from_existing_index(
        index_name="llmdemo", embedding=embedding
    )
    retriever = docsearch.as_retriever()

    # LLM for completing prompts, and other tasks.
    llm = OpenAI(temperature=0, max_tokens=128)

    # Conversation memory.
    memory = ConversationSummaryBufferMemory(
        max_token_limit=650,
        llm=llm,
        memory_key="chat_history",
        output_key='answer'
    )

    # Conversational chain puts it all together.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
        get_chat_history=identity,
        max_tokens_limit=4096
    )

    # Language mismatch fix:
    chain.combine_docs_chain.llm_chain.prompt.template = \
        "Use the following pieces of context to answer the question at the end " \
        "in the same language as the question. If you don't know the answer, " \
        "just say that you don't know, don't try to make up an answer.\n\n" \
        "{context}\n\n" \
        "Question: {question}\n" \
        "Helpful Answer: "

    # Trulens instrumentation.
    tc = tru_chain.TruChain(chain, chain_id=chain_id)

    return tc(prompt)


# Set up Streamlit app
st.title("TruBot")
user_input = st.text_input("Ask a question")

if user_input:
    # Generate GPT-3 response
    prompt_input = user_input
    # add context manager to capture tokens and cost of the chain

    with get_openai_callback() as cb:
        response, record = generate_response(prompt_input)
        total_tokens = cb.total_tokens
        total_cost = cb.total_cost

    answer = response['answer']

    # Display response
    st.write("Here's some help for you:")
    st.write(answer)

    record_id = tru.add_data(
        chain_id=chain_id,
        prompt=prompt_input,
        response=answer,
        details=record,
        tags='dev',
        total_tokens=total_tokens,
        total_cost=total_cost
    )

    # Run feedback function and get value
    """
    feedback = tru.run_feedback_function(
        prompt_input, answer, [
            tru_feedback.get_not_hate_function(
                evaluation_choice='prompt',
                provider='openai',
                model_engine='moderation'
            ),
            tru_feedback.get_sentimentpositive_function(
                evaluation_choice='response',
                provider='openai',
                model_engine='gpt-3.5-turbo'
            ),
            tru_feedback.get_relevance_function(
                evaluation_choice='both',
                provider='openai',
                model_engine='gpt-3.5-turbo'
            )
        ]
    )

    # Add value to database
    tru.add_feedback(record_id, feedback)
    """
