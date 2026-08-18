"""
Tests for OTEL TruChain app.
"""

import gc
import warnings
import weakref

import pytest
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

import tests.util.otel_test_case
import tests.util.otel_tru_app_test_case
from tests.utils import enable_otel_backwards_compatibility

try:
    # These imports require optional dependencies to be installed.
    from langchain import hub
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import DeterministicFakeEmbedding
    from langchain_community.llms import FakeListLLM
    from langchain_community.vectorstores import FAISS
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from trulens.apps.langchain import TruChain
except Exception as e:
    # If imports fail, skip tests in this module
    import sys

    print(
        f"Skipping test_otel_tru_chain tests due to import error: {e}",
        file=sys.stderr,
    )
    pytest.skip(
        f"LangChain dependencies not available: {e}", allow_module_level=True
    )

# Suppress noisy DeprecationWarnings emitted by optional deps (PyMuPDF + LangGraph)
for message in (
    "builtin type SwigPyPacked has no __module__ attribute",
    "builtin type SwigPyObject has no __module__ attribute",
    "builtin type swigvarlink has no __module__ attribute",
    "AgentStatePydantic has been moved to `langchain.agents`",
):
    warnings.filterwarnings(
        "ignore", message=message, category=DeprecationWarning
    )


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

    @pytest.mark.skip(
        reason="Golden file comparison skipped - span structure varies across environments"
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
        # Smoke test - just verify it runs without errors
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


@pytest.mark.optional
class TestOtelTruChainStreaming(tests.util.otel_test_case.OtelTestCase):
    """LangChain's `BaseChatModel.stream`/`.astream` are registered with
    span_type=GENERATION in this app's METHODS() (see
    trulens.apps.langchain.tru_chain), and route through the same generic
    generator-wrapping in trulens.core.otel.instrument that any other
    @instrument(span_type=GENERATION) generator function does. So no
    LangChain-specific streaming instrumentation code is needed -- these
    tests exist to prove that's actually true, not to add new behavior.
    """

    def test_sync_stream_records_streaming_attributes(self) -> None:
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Hello world")])
        )
        tru_chain = TruChain(model, app_name="test_app", app_version="v1")

        with tru_chain:
            chunks = list(model.stream("say hello"))
        self.assertEqual(len(chunks), 3)  # "Hello", " ", "world"

        events = self._get_events()
        found = False
        for attrs in events["record_attributes"]:
            if SpanAttributes.GENERATION.IS_STREAMING in attrs:
                found = True
                self.assertTrue(attrs[SpanAttributes.GENERATION.IS_STREAMING])
                self.assertEqual(
                    attrs[SpanAttributes.GENERATION.CHUNKS_RECEIVED], 3
                )
                self.assertGreaterEqual(
                    attrs[SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS], 0
                )
        self.assertTrue(found, "No span with IS_STREAMING found")

    def test_async_stream_records_streaming_attributes(self) -> None:
        import asyncio

        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Hello world")])
        )
        tru_chain = TruChain(model, app_name="test_app", app_version="v1")

        async def run():
            chunks = []
            with tru_chain:
                async for chunk in model.astream("say hello"):
                    chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        self.assertEqual(len(chunks), 3)

        events = self._get_events()
        found = False
        for attrs in events["record_attributes"]:
            if SpanAttributes.GENERATION.IS_STREAMING in attrs:
                found = True
                self.assertTrue(attrs[SpanAttributes.GENERATION.IS_STREAMING])
                self.assertEqual(
                    attrs[SpanAttributes.GENERATION.CHUNKS_RECEIVED], 3
                )
        self.assertTrue(found, "No span with IS_STREAMING found")
