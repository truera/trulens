import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import Pinecone
import numpy as np
import pinecone
import streamlit as st

from trulens_eval import tru
from trulens_eval import tru_chain
from trulens_eval import tru_feedback
from trulens_eval.keys import *
from trulens_eval.keys import PINECONE_API_KEY
from trulens_eval.keys import PINECONE_ENV
from trulens_eval.tru_db import Record
from trulens_eval.tru_feedback import Feedback

# Set up GPT-3 model
model_name = "gpt-3.5-turbo"

chain_id = "TruBot"
# chain_id = "TruBot_langprompt"
# chain_id = "TruBot_relevance"

# Pinecone configuration.
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)

identity = lambda h: h

hugs = tru_feedback.Huggingface()
openai = tru_feedback.OpenAI()

f_lang_match = Feedback(hugs.language_match).on(
    text1="prompt", text2="response"
)

f_qa_relevance = Feedback(openai.relevance).on(
    prompt="input", response="output"
)

f_qs_relevance = Feedback(openai.qs_relevance).on(
    question="input",
    statement=Record.chain.combine_docs_chain._call.args.inputs.input_documents
).on_multiple(
    multiarg="statement", each_query=Record.page_content, agg=np.min
)


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
    if "langprompt" in chain_id:
        chain.combine_docs_chain.llm_chain.prompt.template = \
            "Use the following pieces of CONTEXT to answer the question at the end " \
            "in the same language as the question. If you don't know the answer, " \
            "just say that you don't know, don't try to make up an answer.\n" \
            "\n" \
            "CONTEXT: {context}\n" \
            "\n" \
            "Question: {question}\n" \
            "Helpful Answer: "

    elif "relevance" in chain_id:
        # Contexts fix
        chain.combine_docs_chain.llm_chain.prompt.template = \
            "Use only the relevant contexts to answer the question at the end " \
            ". Some pieces of context may not be relevant. If you don't know the answer, " \
            "just say that you don't know, don't try to make up an answer.\n" \
            "\n" \
            "Contexts: \n" \
            "{context}\n" \
            "\n" \
            "Question: {question}\n" \
            "Helpful Answer: "

        # space is important

        chain.combine_docs_chain.document_prompt.template = "\tContext: {page_content}"

    # Trulens instrumentation.
    tc = tru_chain.TruChain(chain, chain_id=chain_id)

    return tc, tc.call_with_record(dict(question=prompt))


# Set up Streamlit app
st.title("TruBot")
user_input = st.text_input("Ask a question about TruEra")

if user_input:
    # Generate GPT-3 response
    prompt_input = user_input
    # add context manager to capture tokens and cost of the chain

    with get_openai_callback() as cb:
        chain, (response, record) = generate_response(prompt_input)
        total_tokens = cb.total_tokens
        total_cost = cb.total_cost

    answer = response['answer']

    # Display response
    st.write(answer)

    record_id = tru.add_data(
        chain_id=chain_id,
        prompt=prompt_input,
        response=answer,
        record=record,
        tags='dev',
        total_tokens=total_tokens,
        total_cost=total_cost
    )

    # Run feedback function and get value
    feedbacks = tru.run_feedback_functions(
        chain=chain,
        record=record,
        feedback_functions=[f_lang_match, f_qa_relevance, f_qs_relevance]
    )

    # Add value to database
    tru.add_feedback(record_id, feedbacks)
