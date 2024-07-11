import json
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import \
    Pinecone as PineconeVectorStore
from snowflake.core import Root
from snowflake.snowpark import Session
import streamlit as st


class PineconeRetriever:

    def __init__(self):
        self.embed_model = HuggingFaceEmbeddings(
            model_name="Snowflake/snowflake-arctic-embed-m"
        )
        self.index_name = "streamlit-docs"

    def retrieve(self, query: str):
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embed_model,
            text_key="_node_content"
        )
        retriever = docsearch.as_retriever()
        nodes = retriever.invoke(query)
        contents = [json.loads(t.page_content) for t in nodes]
        texts = [tc.get("text") for tc in contents]
        return texts


class CortexSearchRetriever:

    def __init__(self, limit_to_retrieve: int = 4):
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str):
        connection_parameters = {
            "account": os.environ["TRULENS_SNOWFLAKE_ACCOUNT"],
            "user": os.environ["TRULENS_SNOWFLAKE_USER"],
            "password": os.environ["TRULENS_SNOWFLAKE_USER_PASSWORD"],
            "role": os.environ["TRULENS_SNOWFLAKE_ROLE"],
            "warehouse": os.environ["TRULENS_SNOWFLAKE_WAREHOUSE"],
        }
        session = None
        try:
            session = Session.builder.configs(connection_parameters).create()
            root = Root(session)
            cortex_search_service = root.databases[
                os.environ["TRULENS_SNOWFLAKE_DATABASE"]].schemas[
                    os.environ["TRULENS_SNOWFLAKE_SCHEMA"]].cortex_search_services[
                        os.environ["TRULENS_SNOWFLAKE_CORTEX_SEARCH_SERVICE"]]
            resp = cortex_search_service.search(
                query=query,
                columns=["doc_text"],
                limit=self._limit_to_retrieve,
            )
            if resp.results:
                return [curr["doc_text"] for curr in resp.results]
            return []
        finally:
            try:
                if session is not None:
                    session.close()
            except:
                pass


AVAILABLE_RETRIEVERS = [
    "Cortex Search",
    "Pinecone",
]


@st.cache_resource
def get_retriever(retriever_name: str):
    if retriever_name not in AVAILABLE_RETRIEVERS:
        raise ValueError(f"Retriever {retriever_name} not available.")
    elif retriever_name == "Cortex Search":
        return CortexSearchRetriever()
    elif retriever_name == "Pinecone":
        return PineconeRetriever()
