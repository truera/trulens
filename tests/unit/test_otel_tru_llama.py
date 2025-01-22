"""
Tests for OTEL TruLlama app.
"""

from unittest import main

from trulens.core.session import TruSession

from tests.test import optional_test
from tests.test import run_optional_tests
from tests.util.otel_app_test_case import OtelAppTestCase

if run_optional_tests():
    # These imports require optional dependencies to be installed.
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import VectorStoreIndex
    from llama_index.core.llms.mock import MockLLM
    from trulens.apps.llamaindex import TruLlama


@optional_test
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
        index = VectorStoreIndex.from_documents(documents)
        return index.as_query_engine(similarity_top_k=3)

    def test_smoke(self) -> None:
        # TODO(this_pr): extract this out with tru_chain testing.
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create app.
        rag = self._create_simple_rag()
        tru_recorder = TruLlama(
            rag,
            app_name="Simple RAG",
            app_version="v1",
        )
        # Record and invoke.
        with tru_recorder(run_name="test run", input_id="42"):
            rag.query("What is multi-headed attention?")
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_llama__test_smoke.csv"
        )


if __name__ == "__main__":
    main()
