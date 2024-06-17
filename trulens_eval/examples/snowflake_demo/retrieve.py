import json
import os

from dotenv import load_dotenv
from feedback import f_context_relevance
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import \
    Pinecone as PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

from trulens_eval.utils.langchain import WithFeedbackFilterDocuments

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
        filtered_retriever = WithFeedbackFilterDocuments.of_retriever(
            retriever=retriever, feedback=f_context_relevance, threshold=0
        )
        nodes = filtered_retriever.invoke(query)
        contents = [json.loads(t.page_content) for t in nodes]
        texts = [tc.get("text") for tc in contents]
        return texts
