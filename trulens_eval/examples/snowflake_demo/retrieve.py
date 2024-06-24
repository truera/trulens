import json
import os

from dotenv import load_dotenv
from feedback import f_context_relevance
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import \
    Pinecone as PineconeVectorStore
from snowflake.core import Root
from snowflake.snowpark import Session

load_dotenv()


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
        connection_parameters = {
            "account": os.environ["SF_ACCOUNT"],
            "user": os.environ["SF_USER"],
            "password": os.environ["SF_PASSWORD"],
            "role": os.environ["SF_ROLE"],
            "warehouse": os.environ["SF_WAREHOUSE"],
        }
        session = Session.builder.configs(connection_parameters).create()
        root = Root(session)
        self._cortex_search_service = root.databases[
            os.environ["SF_DB_NAME"]].schemas[
                os.environ["SF_SCHEMA"]].cortex_search_services[
                    os.environ["SF_CORTEX_SEARCH_SERVICE"]]

    def retrieve(self, query: str):
        resp = self._cortex_search_service.search(
            query=query,
            columns=["doc_text"],
            limit=self._limit_to_retrieve,
        )
        if resp.results:
            return [curr["doc_text"] for curr in resp.results]
        return []


AVAILABLE_RETRIEVERS = {
    "Cortex Search": CortexSearchRetriever(),
    "Pinecone": PineconeRetriever(),
}
