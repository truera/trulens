"""
Tests for OTEL TruChain app.
"""

import pytest
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession

from tests.util.otel_app_test_case import OtelAppTestCase

try:
    # These imports require optional dependencies to be installed.
    from langchain import hub
    from langchain.schema import StrOutputParser
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import DeterministicFakeEmbedding
    from langchain_community.llms import FakeListLLM
    from langchain_community.vectorstores import FAISS
    from langchain_core.runnables import RunnablePassthrough
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from trulens.apps.langchain import TruChain
except Exception:
    pass


@pytest.mark.optional
class TestOtelTruChain(OtelAppTestCase):
    @staticmethod
    def _create_simple_rag():
        # Helper function.
        @instrument(attributes={"best_baby": "Kojikun"})
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create documents.
        loader = PyPDFLoader("./tests/unit/data/attention_is_all_you_need.pdf")
        docs = loader.load_and_split()
        # Create vector store.
        embeddings = DeterministicFakeEmbedding(size=10)
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents, embeddings)
        # Create RAG.
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        llm = FakeListLLM(
            responses=[
                f"This is a mocked response for prompt {i}." for i in range(100)
            ]
        )
        return (
            {
                "question": RunnablePassthrough(),
                "context": retriever | format_docs,
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def test_missing_main_method_raises_error(self):
        # Attempt to create a TruLlama without specifying main_method.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create app.
        rag_chain = self._create_simple_rag()
        with self.assertRaises(ValueError) as context:
            TruChain(rag_chain, app_name="Simple RAG", app_version="v1")

        self.assertIn("main_method", str(context.exception))

    def test_smoke(self) -> None:
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create app.
        rag_chain = self._create_simple_rag()
        tru_recorder = TruChain(
            rag_chain,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag_chain.invoke,
        )
        # Record and invoke.
        with tru_recorder(run_name="test run", input_id="42"):
            rag_chain.invoke("What is multi-headed attention?")
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_chain__test_smoke.csv"
        )
