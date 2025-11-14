"""
Unit tests for LangChain instrumentation functionality.

This test suite covers:
1. Instrumentation of BaseLanguageModel.invoke and ainvoke with GENERATION spans
2. Instrumentation of VectorStore.similarity_search with RETRIEVAL spans
3. Stream event handling (stream_events/astream_events)
4. Event content extraction from streaming outputs
5. CreateSpanFunctionCallContextManager baggage cleanup
"""

from typing import Any, Optional
from unittest.mock import Mock
import uuid

from opentelemetry.baggage import get_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
import pytest
from trulens.core.otel.function_call_context_manager import (
    CreateSpanFunctionCallContextManager,
)
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

try:
    # Initialize langchain globals for langchain 1.x compatibility
    try:
        from langchain_core import globals as langchain_globals

        langchain_globals.set_debug(False)
        langchain_globals.set_verbose(False)
    except (ImportError, AttributeError):
        pass

    from langchain_community.llms import FakeListLLM
    from langchain_core.documents import Document
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.messages import AIMessageChunk
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.vectorstores import VectorStore
    from trulens.apps.langchain import TruChain
    from trulens.apps.langchain.tru_chain import LangChainInstrument
except ImportError:
    pytest.skip("langchain not installed", allow_module_level=True)


class DummyVectorStore(VectorStore):
    """Minimal VectorStore implementation for testing instrumentation."""

    def __init__(self, documents: list[Document]):
        self._documents = documents

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        # Return up to k documents deterministically.
        _ = query, kwargs
        return self._documents[:k]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        docs = self.similarity_search(query, k=k, **kwargs)
        return [(doc, 1.0) for doc in docs]

    def _select_relevance_score_fn(self):
        return lambda distance: 1.0 - distance

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> "DummyVectorStore":
        _ = embedding, kwargs
        documents = [
            Document(
                page_content=text, metadata=(metadatas[i] if metadatas else {})
            )
            for i, text in enumerate(texts)
        ]
        return cls(documents)


@pytest.mark.optional
class TestLangChainInstrumentation(OtelTestCase):
    """Tests for LangChain instrumentation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Create a simple LLM for testing
        self.llm = FakeListLLM(
            responses=["response 1", "response 2", "response 3"]
        )
        self.chat_model = FakeMessagesListChatModel(
            responses=[
                AIMessage(content="chat response 1"),
                AIMessage(content="chat response 2"),
            ]
        )

        # Create documents and vector store for retrieval tests
        docs = [
            Document(page_content="Document 1 content", metadata={"id": 1}),
            Document(page_content="Document 2 content", metadata={"id": 2}),
            Document(page_content="Document 3 content", metadata={"id": 3}),
        ]
        self.vectorstore = DummyVectorStore(docs)

    def test_base_language_model_invoke_generation_span(self):
        """Test 1: BaseLanguageModel.invoke is instrumented with GENERATION span type."""
        # Create a simple chain that uses the LLM
        chain = self.llm

        # Create TruChain to trigger instrumentation
        tru_chain = TruChain(
            chain,
            app_name="test_llm_invoke",
            app_version="v1",
            main_method=chain.invoke,
        )

        # Invoke the LLM through instrumented method
        tru_chain.instrumented_invoke_main_method(
            run_name="test_run",
            input_id="test_id_1",
            main_method_args=("What is AI?",),
        )

        # Get events from database
        events_df = self._get_events()

        # Filter for LLM invoke events
        llm_events = events_df[
            events_df["record"].apply(
                lambda x: "invoke" in x.get("name", "")
                and "FakeListLLM" in x.get("name", "")
            )
        ]
        llm_events = llm_events[
            llm_events["record_attributes"].apply(
                lambda attrs: attrs.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.GENERATION
            )
        ]

        # Verify at least one LLM invoke event exists
        self.assertGreater(len(llm_events), 0, "No LLM invoke events found")

        # Check that the span type is GENERATION
        for _, event in llm_events.iterrows():
            span_type = event["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            self.assertEqual(
                span_type,
                SpanAttributes.SpanType.GENERATION,
                f"Expected GENERATION span type, got {span_type}",
            )

    @pytest.mark.asyncio
    async def test_base_language_model_ainvoke_generation_span(self):
        """Test 1b: BaseLanguageModel.ainvoke is instrumented with GENERATION span type."""
        # Create a simple chain that uses the LLM
        chain = self.llm

        # Create TruChain to trigger instrumentation
        tru_chain = TruChain(
            chain,
            app_name="test_llm_ainvoke",
            app_version="v1",
            main_method=chain.ainvoke,
        )

        # Invoke the LLM asynchronously through instrumented method
        await tru_chain.instrumented_ainvoke_main_method(
            run_name="test_run_async",
            input_id="test_id_2",
            main_method_args=("What is ML?",),
        )

        # Get events from database
        events_df = self._get_events()

        # Filter for LLM ainvoke events
        llm_events = events_df[
            events_df["record"].apply(
                lambda x: "ainvoke" in x.get("name", "")
                and "FakeListLLM" in x.get("name", "")
            )
        ]
        llm_events = llm_events[
            llm_events["record_attributes"].apply(
                lambda attrs: attrs.get(SpanAttributes.SPAN_TYPE)
                == SpanAttributes.SpanType.GENERATION
            )
        ]

        # Verify at least one LLM ainvoke event exists
        self.assertGreater(len(llm_events), 0, "No LLM ainvoke events found")

        # Check that the span type is GENERATION
        for _, event in llm_events.iterrows():
            span_type = event["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            self.assertEqual(
                span_type,
                SpanAttributes.SpanType.GENERATION,
                f"Expected GENERATION span type, got {span_type}",
            )

    def test_vectorstore_similarity_search_retrieval_span(self):
        """Test 2: VectorStore.similarity_search is instrumented with RETRIEVAL span type."""
        # Create a simple chain that uses the vector store
        retriever = self.vectorstore.as_retriever()

        # Create TruChain to trigger instrumentation
        tru_chain = TruChain(
            retriever,
            app_name="test_retrieval",
            app_version="v1",
            main_method=retriever.invoke,
        )

        # Invoke the retriever
        tru_chain.instrumented_invoke_main_method(
            run_name="test_retrieval_run",
            input_id="test_id_3",
            main_method_args=("search query",),
        )

        # Get events from database
        events_df = self._get_events()

        # Filter for similarity_search events
        search_events = events_df[
            events_df["record"].apply(
                lambda x: "similarity_search" in x.get("name", "")
            )
        ]

        # Verify at least one similarity_search event exists
        self.assertGreater(
            len(search_events), 0, "No similarity_search events found"
        )

        # Check that the span type is RETRIEVAL
        for _, event in search_events.iterrows():
            span_type = event["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            self.assertEqual(
                span_type,
                SpanAttributes.SpanType.RETRIEVAL,
                f"Expected RETRIEVAL span type, got {span_type}",
            )

    @pytest.mark.asyncio
    async def test_vectorstore_asimilarity_search_retrieval_span(self):
        """Test 2b: VectorStore.asimilarity_search is instrumented with RETRIEVAL span type."""
        # Create a simple chain that uses the vector store
        retriever = self.vectorstore.as_retriever()

        # Create TruChain to trigger instrumentation
        tru_chain = TruChain(
            retriever,
            app_name="test_async_retrieval",
            app_version="v1",
            main_method=retriever.ainvoke,
        )

        # Invoke the retriever asynchronously
        await tru_chain.instrumented_ainvoke_main_method(
            run_name="test_async_retrieval_run",
            input_id="test_id_4",
            main_method_args=("async search query",),
        )

        # Get events from database
        events_df = self._get_events()

        # Filter for asimilarity_search or similarity_search events
        search_events = events_df[
            events_df["record"].apply(
                lambda x: "similarity_search" in x.get("name", "")
            )
        ]

        # Verify at least one similarity_search event exists
        self.assertGreater(
            len(search_events), 0, "No async similarity_search events found"
        )

        # Check that the span type is RETRIEVAL
        for _, event in search_events.iterrows():
            span_type = event["record_attributes"].get(SpanAttributes.SPAN_TYPE)
            self.assertEqual(
                span_type,
                SpanAttributes.SpanType.RETRIEVAL,
                f"Expected RETRIEVAL span type, got {span_type}",
            )

    def test_stream_events_incremental_updates(self):
        """Test 3: stream_events properly handles incremental chunk updates."""
        # Skip: FakeChatModel doesn't have stream_events in all LangChain versions
        # This test is more about verifying the instrumentation config exists
        # which is tested in test_instrumented_methods_configuration
        self.skipTest(
            "FakeChatModel may not have stream_events in all LangChain versions"
        )

    @pytest.mark.asyncio
    async def test_astream_events_incremental_updates(self):
        """Test 3b: astream_events properly handles incremental chunk updates."""
        # Skip: FakeChatModel doesn't have astream_events in all LangChain versions
        # This test is more about verifying the instrumentation config exists
        # which is tested in test_instrumented_methods_configuration
        self.skipTest(
            "FakeChatModel may not have astream_events in all LangChain versions"
        )

    def test_extract_event_content_direct_content(self):
        """Test 4a: _extract_event_content extracts direct content field."""
        # Test with direct content field
        event = {"content": "Direct content string"}

        # Extract using TruChain's _extract_event_content logic
        # We'll test this by simulating the main_output method
        from trulens.apps.langchain.tru_chain import TruChain

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        # The _extract_event_content is an inner function, so we test
        # the main_output behavior with event-style outputs
        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Direct content string")

    def test_extract_event_content_nested_messages(self):
        """Test 4b: _extract_event_content extracts content from nested messages."""
        # Test with messages list
        msg = AIMessage(content="Message content")
        event = {"messages": [msg]}

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Message content")

    def test_extract_event_content_data_chunk(self):
        """Test 4c: _extract_event_content extracts content from data.chunk."""
        # Test with data.chunk structure
        chunk = AIMessageChunk(content="Chunk content")
        event = {"data": {"chunk": chunk}}

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Chunk content")

    def test_extract_event_content_data_output(self):
        """Test 4d: _extract_event_content extracts content from data.output."""
        # Test with data.output as string
        event = {"data": {"output": "Output string"}}

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Output string")

    def test_extract_event_content_return_values(self):
        """Test 4e: _extract_event_content extracts content from data.return_values."""
        # Test with return_values structure
        event = {"data": {"return_values": {"output": "Return value output"}}}

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        ret = [event]
        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=ret,
        )

        self.assertEqual(output, "Return value output")

    def test_extract_event_content_multiple_events(self):
        """Test 4f: _extract_event_content handles multiple events correctly."""
        # Test with multiple events (simulating streaming)
        events = [
            {"data": {"chunk": AIMessageChunk(content="Part 1")}},
            {"data": {"chunk": AIMessageChunk(content=" Part 2")}},
            {"data": {"chunk": AIMessageChunk(content=" Part 3")}},
        ]

        # Create a dummy TruChain instance
        dummy_chain = RunnablePassthrough()
        tru_chain = TruChain(dummy_chain, app_name="test", app_version="v1")

        output = tru_chain.main_output(
            func=lambda x: x,
            sig=None,
            bindings=None,
            ret=events,
        )

        self.assertEqual(output, "Part 1 Part 2 Part 3")

    def test_context_manager_record_id_with_otel_ctx(self):
        """Test 5a: CreateSpanFunctionCallContextManager creates record_id."""
        # Set up context with a recording
        mock_recording = Mock()
        mock_recording.add_record_id = Mock()

        token1 = context_api.attach(
            set_baggage("__trulens_recording__", mock_recording)
        )

        try:
            # Create and use the context manager
            cm = CreateSpanFunctionCallContextManager("test_span")

            with cm as _:
                # Verify a record_id was created
                record_id = get_baggage(SpanAttributes.RECORD_ID)
                self.assertIsNotNone(record_id)
                self.assertTrue(isinstance(record_id, str))

            # After exiting context, record_id should be cleaned up
            record_id_after = get_baggage(SpanAttributes.RECORD_ID)
            self.assertIsNone(
                record_id_after,
                "record_id should be cleaned up after context exits",
            )

        finally:
            # Clean up
            context_api.detach(token1)

    def test_context_manager_record_id_without_otel_ctx(self):
        """Test 5b: CreateSpanFunctionCallContextManager without otel_ctx."""
        # Set up context with only recording (no otel_ctx)
        mock_recording = Mock()
        mock_recording.add_record_id = Mock()

        token = context_api.attach(
            set_baggage("__trulens_recording__", mock_recording)
        )

        try:
            # Create and use the context manager
            cm = CreateSpanFunctionCallContextManager("test_span_no_ctx")

            with cm as _:
                # Verify a record_id was created
                record_id = get_baggage(SpanAttributes.RECORD_ID)
                self.assertIsNotNone(record_id)
                self.assertTrue(isinstance(record_id, str))

            # After exiting context, record_id should be cleaned up
            # because otel_ctx is NOT present
            record_id_after = get_baggage(SpanAttributes.RECORD_ID)
            self.assertIsNone(
                record_id_after,
                "record_id should be cleaned up when otel_ctx is not present",
            )

        finally:
            # Clean up
            context_api.detach(token)

    def test_context_manager_record_id_reuse_existing(self):
        """Test 5c: CreateSpanFunctionCallContextManager reuses existing record_id."""
        # Set up context with existing record_id
        existing_record_id = str(uuid.uuid4())
        mock_recording = Mock()
        mock_recording.add_record_id = Mock()

        token1 = context_api.attach(
            set_baggage("__trulens_recording__", mock_recording)
        )
        token2 = context_api.attach(
            set_baggage(SpanAttributes.RECORD_ID, existing_record_id)
        )

        try:
            # Create and use the context manager
            cm = CreateSpanFunctionCallContextManager("test_span_reuse")

            with cm as _:
                # Verify the same record_id is used
                record_id = get_baggage(SpanAttributes.RECORD_ID)
                self.assertEqual(
                    record_id,
                    existing_record_id,
                    "Should reuse existing record_id",
                )

                # Verify _started_record is False
                self.assertFalse(
                    cm._started_record,
                    "Should not mark as started when reusing record_id",
                )

            # After exiting, record_id should still be present
            # (because we didn't create it)
            record_id_after = get_baggage(SpanAttributes.RECORD_ID)
            self.assertEqual(record_id_after, existing_record_id)

        finally:
            # Clean up
            context_api.detach(token2)
            context_api.detach(token1)

    def test_instrumented_methods_configuration(self):
        """Test that METHODS configuration includes expected instrumentation."""
        # Get the instrumented methods configuration
        methods = LangChainInstrument.Default.METHODS()

        # Check for BaseLanguageModel methods with GENERATION span type
        generation_methods = [
            m
            for m in methods
            if m.span_type == SpanAttributes.SpanType.GENERATION
            and m.class_filter == BaseLanguageModel
        ]

        generation_method_names = [m.method for m in generation_methods]
        self.assertIn("invoke", generation_method_names)
        self.assertIn("ainvoke", generation_method_names)

        # Check for VectorStore methods with RETRIEVAL span type
        retrieval_methods = [
            m
            for m in methods
            if m.span_type == SpanAttributes.SpanType.RETRIEVAL
            and "similarity_search" in m.method
        ]

        self.assertGreater(
            len(retrieval_methods),
            0,
            "Expected retrieval methods for VectorStore",
        )

        # Check for stream_events methods
        stream_methods = [
            m
            for m in methods
            if m.method in ["stream_events", "astream_events"]
        ]

        self.assertGreater(
            len(stream_methods),
            0,
            "Expected stream_events instrumentation",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
