"""
Tests for OTEL TruLlama app.
"""

import pytest
from trulens.otel.semconv.constants import (
    TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG,
)

from tests.util.otel_app_test_case import OtelAppTestCase

try:
    # These imports require optional dependencies to be installed.
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex
    from llama_index.core.llms.mock import MockLLM
    from trulens.apps.llamaindex import TruLlama

    from tests.util.llama_index_mock_embedder import MockEmbedding
except Exception:
    pass


_UUID_REGEX = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_FLOAT_REGEX = r"[-]?\d+\.\d+"
_CONTEXT_RETRIEVAL_REGEX = (
    r"Node ID: "
    + _UUID_REGEX
    + r"\n([\s\S]*?)\nScore:\s*"
    + _FLOAT_REGEX
    + "\n"
)
_CONTEXT_RETRIEVAL_REPLACEMENT = r"\1"


@pytest.mark.optional
class TestOtelTruLlama(OtelAppTestCase):
    @staticmethod
    def _create_simple_rag():
        Settings.chunk_size = 128
        Settings.chunk_overlap = 16
        Settings.llm = MockLLM()
        reader = SimpleDirectoryReader(
            input_files=["./tests/unit/data/attention_is_all_you_need.pdf"],
        )
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(
            documents, embed_model=MockEmbedding(10)
        )
        return index.as_query_engine(similarity_top_k=3)

    def test_missing_main_method_raises_error(self):
        # Create app.
        rag = self._create_simple_rag()
        with self.assertRaises(ValueError) as context:
            TruLlama(rag, app_name="Simple RAG", app_version="v1")

        self.assertIn("main_method", str(context.exception))

    def test_smoke(self) -> None:
        # Create app.
        rag = self._create_simple_rag()
        tru_recorder = TruLlama(
            rag,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag.query,
        )
        # Record and invoke.
        tru_recorder.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("What is multi-headed attention?",),
        )
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_llama__test_smoke.csv",
            regex_replacements=[
                # This changes [Node ID <UUID>: <TEXT_WE_WANT> Score: <SCORE>]
                # strings to just <TEXT_WE_WANT>. We don't want the SCORE
                # due to precision issues causing it to be slightly different
                # in some runs.
                (_CONTEXT_RETRIEVAL_REGEX, _CONTEXT_RETRIEVAL_REPLACEMENT)
            ],
        )

    def test_app_specific_record_root(self) -> None:
        rag1 = self._create_simple_rag()
        rag2 = self._create_simple_rag()
        TruLlama(
            rag1,
            app_name="Simple RAG",
            app_version="v1",
            main_method=rag1.query,
        )
        rag3 = self._create_simple_rag()

        def count_wraps(func):
            if not hasattr(func, "__wrapped__"):
                return 0
            return 1 + count_wraps(func.__wrapped__)

        self.assertEqual(count_wraps(rag1.query), 2)
        self.assertEqual(count_wraps(rag2.query), 1)
        self.assertEqual(count_wraps(rag3.query), 1)
        self.assertFalse(
            hasattr(rag1.query, TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG)
        )
        self.assertFalse(
            hasattr(rag2.query, TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG)
        )
        self.assertFalse(
            hasattr(rag3.query, TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG)
        )
