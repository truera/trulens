"""
Tests for OTEL TruChain app.
"""

from unittest import main

from langchain import hub
from langchain.embeddings.fake import DeterministicFakeEmbedding
from langchain.llms.fake import FakeListLLM
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trulens.apps.langchain import TruChain
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.init import init
from trulens.experimental.otel_tracing.core.instrument import instrument

from tests.util.df_comparison import (
    compare_dfs_accounting_for_ids_and_timestamps,
)
from tests.util.otel_app_test_case import OtelAppTestCase


class TestOtelTruChain(OtelAppTestCase):
    @staticmethod
    def _create_simple_rag():
        # Helper function.
        @instrument(attributes={"best_baby": "kojikun"})
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create documents.
        loader = PyPDFLoader(
            # TODO(this_pr): Remove hardcoded path.
            "/Users/dkurokawa/Work/code/trulens/trulens/tests/unit/data/attention_is_all_you_need.pdf"
        )
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
        init(tru_session, debug=True)
        # Create app.
        rag_chain = self._create_simple_rag()
        tru_recorder = TruChain(
            rag_chain,
            app_name="Simple RAG",
            app_version="v1",
        )
        # Record and invoke.
        with tru_recorder:
            rag_chain.invoke("What is multi-headed attention?")
        # Compare results to expected.
        GOLDEN_FILENAME = (
            "tests/unit/static/golden/test_otel_tru_chain__test_smoke.csv"
        )
        actual = self._get_events()
        self.assertEqual(len(actual), 13)
        self.write_golden(GOLDEN_FILENAME, actual)
        expected = self.load_golden(GOLDEN_FILENAME)
        self._convert_column_types(expected)
        compare_dfs_accounting_for_ids_and_timestamps(
            self,
            expected,
            actual,
            ignore_locators=[
                f"df.iloc[{i}][resource_attributes][telemetry.sdk.version]"
                for i in range(10)
            ],
        )


if __name__ == "__main__":
    main()
