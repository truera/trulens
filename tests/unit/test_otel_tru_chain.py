"""
Tests for OTEL TruChain app.
"""

from unittest import main

import pandas as pd
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.instrument import instrument

from tests.test import optional_test
from tests.test import run_optional_tests
from tests.util.df_comparison import (
    compare_dfs_accounting_for_ids_and_timestamps,
)
from tests.util.otel_app_test_case import OtelAppTestCase

if run_optional_tests():
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


@optional_test
class TestOtelTruChain(OtelAppTestCase):
    @staticmethod
    def _create_simple_rag():
        # Helper function.
        @instrument(attributes={"best_baby": "kojikun"})
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
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

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
        )
        # Record and invoke.
        with tru_recorder(run_name="test run", input_id="42"):
            rag_chain.invoke("What is multi-headed attention?")
        # Compare results to expected.
        GOLDEN_FILENAME = (
            "tests/unit/static/golden/test_otel_tru_chain__test_smoke.csv"
        )
        tru_session.experimental_force_flush()
        actual = self._get_events()
        self.write_golden(GOLDEN_FILENAME, actual)
        expected = self.load_golden(GOLDEN_FILENAME)
        self._convert_column_types(expected)
        compare_dfs_accounting_for_ids_and_timestamps(
            self,
            expected,
            actual,
            ignore_locators=[
                f"df.iloc[{i}][resource_attributes][telemetry.sdk.version]"
                for i in range(len(expected))
            ],
            timestamp_tol=pd.Timedelta("0.02s"),
        )


if __name__ == "__main__":
    main()
