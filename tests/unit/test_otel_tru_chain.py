"""
Tests for OTEL TruChain app.
"""

import gc
import weakref

import pytest
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

import tests.util.otel_tru_app_test_case
from tests.utils import enable_otel_backwards_compatibility

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
class TestOtelTruChain(tests.util.otel_tru_app_test_case.OtelTruAppTestCase):
    @staticmethod
    def _create_simple_rag():
        # Helper function.
        @instrument(
            attributes=lambda ret, exception, *args, **kwargs: {
                f"{SpanAttributes.UNKNOWN.base}.best_baby": "Kojikun"
            }
        )
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

    @staticmethod
    def _create_test_app_info() -> (
        tests.util.otel_tru_app_test_case.TestAppInfo
    ):
        app = TestOtelTruChain._create_simple_rag()
        return tests.util.otel_tru_app_test_case.TestAppInfo(
            app=app, main_method=app.invoke, TruAppClass=TruChain
        )

    def test_smoke(self) -> None:
        # Create app.
        rag_chain = self._create_simple_rag()
        tru_recorder = TruChain(
            rag_chain,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag_chain.invoke,
        )
        # Record and invoke.
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What is multi-headed attention?",),
        )
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_chain__test_smoke.csv"
        )
        # Check garbage collection.
        # Note that we need to delete `rag_chain` too since `rag_chain` has
        # instrument decorators that have closures of the `tru_recorder` object.
        # Specifically the record root has this at the very least as it calls
        # `TruChain::main_input` for instance.
        tru_recorder_ref = weakref.ref(tru_recorder)
        del tru_recorder
        del rag_chain
        gc.collect()
        self.assertCollected(tru_recorder_ref)

    @enable_otel_backwards_compatibility
    def test_legacy_app(self) -> None:
        # Create app.
        rag_chain = self._create_simple_rag()
        tru_recorder = TruChain(
            rag_chain, app_name="Simple RAG", app_version="v1"
        )
        # Record and invoke.
        with tru_recorder:
            rag_chain.invoke("What is multi-headed attention?")
        # Compare results to expected.
        self._compare_record_attributes_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_chain__test_smoke.csv"
        )
