import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.callbacks import get_openai_callback
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Pinecone
import numpy as np
import pinecone
import streamlit as st
from trulens.core import Select
from trulens.core import Feedback
from trulens.ext.instrument.langchain import TruChain
from trulens.ext.provider.huggingface import Huggingface
from trulens.ext.provider.openai import OpenAI
from trulens.utils.keys import check_keys

check_keys("PINECONE_API_KEY", "PINECONE_ENV", "OPENAI_API_KEY")

# Set up GPT-3 model
model_name = "gpt-3.5-turbo"

app_id = "TruBot"
# app_id = "TruBot_langprompt"
# app_id = "TruBot_relevance"

# Pinecone configuration.
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.environ.get("PINECONE_ENV"),  # next to api key in console
)

identity = lambda h: h

hugs = Huggingface()
openai = OpenAI()

f_lang_match = Feedback(hugs.language_match).on(
    text1=Select.RecordInput, text2=Select.RecordOutput
)

f_qa_relevance = Feedback(openai.relevance).on(
    prompt=Select.RecordInput, response=Select.RecordOutput
)

f_context_relevance = (
    Feedback(openai.context_relevance)
    .on(
        question=Select.RecordInput,
        statement=Select.Record.chain.combine_docs_chain._call.args.inputs.input_documents[
            :
        ].page_content,
    )
    .aggregate(np.min)
)


# @st.cache_data
def generate_response(prompt):
    # Embedding needed for Pinecone vector db.
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")  # 1536 dims
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
        output_key="answer",
    )

    # Conversational chain puts it all together.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
        get_chat_history=identity,
        max_tokens_limit=4096,
    )

    # Language mismatch fix:
    if "langprompt" in app_id:
        chain.combine_docs_chain.llm_chain.prompt.template = (
            "Use the following pieces of CONTEXT to answer the question at the end "
            "in the same language as the question. If you don't know the answer, "
            "just say that you don't know, don't try to make up an answer.\n"
            "\n"
            "CONTEXT: {context}\n"
            "\n"
            "Question: {question}\n"
            "Helpful Answer: "
        )

    elif "relevance" in app_id:
        # Contexts fix
        chain.combine_docs_chain.llm_chain.prompt.template = (
            "Use only the relevant contexts to answer the question at the end "
            ". Some pieces of context may not be relevant. If you don't know the answer, "
            "just say that you don't know, don't try to make up an answer.\n"
            "\n"
            "Contexts: \n"
            "{context}\n"
            "\n"
            "Question: {question}\n"
            "Helpful Answer: "
        )

        # space is important

        chain.combine_docs_chain.document_prompt.template = (
            "\tContext: {page_content}"
        )

    # Trulens instrumentation.
    tc = TruChain(chain, app_id=app_id)

    return tc, tc.with_record(dict(question=prompt))


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

    answer = response["answer"]

    # Display response
    st.write(answer)

    record_id = tru.add_data(
        app_id=app_id,
        prompt=prompt_input,
        response=answer,
        record=record,
        tags="dev",
        total_tokens=total_tokens,
        total_cost=total_cost,
    )

    # Run feedback function and get value
    feedbacks = tru.run_feedback_functions(
        app=app,
        record=record,
        feedback_functions=[f_lang_match, f_qa_relevance, f_context_relevance],
    )

    # Add value to database
    tru.add_feedback(record_id, feedbacks)
