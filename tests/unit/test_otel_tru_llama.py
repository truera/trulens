"""
Tests for OTEL TruLlama app.
"""

import gc
import weakref

import pytest

import tests.util.otel_tru_app_test_case
from tests.utils import enable_otel_backwards_compatibility

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
class TestOtelTruLlama(tests.util.otel_tru_app_test_case.OtelTruAppTestCase):
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

    @staticmethod
    def _create_test_app_info() -> (
        tests.util.otel_tru_app_test_case.TestAppInfo
    ):
        app = TestOtelTruLlama._create_simple_rag()
        return tests.util.otel_tru_app_test_case.TestAppInfo(
            app=app, main_method=app.query, TruAppClass=TruLlama
        )

    @pytest.mark.skip(
        reason="Golden file comparison skipped - span structure changed with langchain 1.x instrumentation improvements"
    )
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
        # Smoke test - just verify it runs without errors
        # Golden file comparison skipped due to span structure changes
        # Check garbage collection.
        # Note that we need to delete `rag` too since `rag` has instrument
        # decorators that have closures of the `tru_recorder` object.
        # Specifically the record root has this at the very least as it calls
        # `TruLlama::main_input` for instance.
        tru_recorder_ref = weakref.ref(tru_recorder)
        del tru_recorder
        del rag
        gc.collect()
        self.assertCollected(tru_recorder_ref)

    @enable_otel_backwards_compatibility
    def test_legacy_app(self) -> None:
        # Create app.
        rag = self._create_simple_rag()
        tru_recorder = TruLlama(rag, app_name="Simple RAG", app_version="v1")
        # Record and invoke.
        with tru_recorder:
            rag.query("What is multi-headed attention?")
        # Compare results to expected.
        self._compare_record_attributes_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_llama__test_smoke.csv"
        )
