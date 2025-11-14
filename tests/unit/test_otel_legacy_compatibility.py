import importlib
import os

import pytest
import trulens.apps.app
from trulens.apps.app import TruApp
from trulens.apps.app import legacy_instrument
from trulens.core.otel.instrument import instrument as otel_instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

try:
    # These imports require optional dependencies to be installed.
    from langchain.chains import LLMChain
    from langchain_community.llms import FakeListLLM
    from langchain_core.prompts import PromptTemplate
    from trulens.apps.langchain import TruChain
except Exception:
    pass


class TestOtelLegacyCompatibility(OtelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Save current state and temporarily disable OTEL before parent setup
        cls._saved_otel_state = os.environ.get("TRULENS_OTEL_TRACING")
        if "TRULENS_OTEL_TRACING" in os.environ:
            del os.environ["TRULENS_OTEL_TRACING"]

        # Set backwards compatibility flag BEFORE calling super() which will enable OTEL
        os.environ["TRULENS_OTEL_BACKWARDS_COMPATIBILITY"] = "1"

        # Now call parent which will set TRULENS_OTEL_TRACING=1
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        if "TRULENS_OTEL_BACKWARDS_COMPATIBILITY" in os.environ:
            del os.environ["TRULENS_OTEL_BACKWARDS_COMPATIBILITY"]
        super().tearDownClass()
        # Restore original state if it existed
        if cls._saved_otel_state is not None:
            os.environ["TRULENS_OTEL_TRACING"] = cls._saved_otel_state
        elif "TRULENS_OTEL_TRACING" in os.environ:
            del os.environ["TRULENS_OTEL_TRACING"]

    def test_import(self) -> None:
        try:
            os.environ["TRULENS_OTEL_TRACING"] = "0"
            importlib.reload(trulens.apps.app)
            from trulens.apps.app import instrument

            self.assertEqual(instrument.__class__, legacy_instrument.__class__)
            os.environ["TRULENS_OTEL_TRACING"] = "1"
            importlib.reload(trulens.apps.app)
            from trulens.apps.app import instrument

            self.assertIsInstance(instrument, otel_instrument)
        finally:
            os.environ["TRULENS_OTEL_TRACING"] = "1"

    def test_legacy_tru_custom_app(self) -> None:
        importlib.reload(trulens.apps.app)
        from trulens.apps.app import instrument

        class TestApp:
            @instrument
            def respond_to_query(self, query: str) -> str:
                self.square(7)
                return f"response to {query}"

            @instrument
            def square(self, n: int) -> int:
                return n * n

        app = TestApp()
        tru_app = TruApp(app, app_name="MyTruCustomApp", app_version="v1")
        with tru_app as recording:
            app.respond_to_query("test")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(2, len(events))
        # Verify first span.
        record_attributes = events["record_attributes"].iloc[0]
        self.assertEqual(
            SpanAttributes.SpanType.RECORD_ROOT,
            record_attributes[SpanAttributes.SPAN_TYPE],
        )
        self.assertEqual(
            "test", record_attributes[SpanAttributes.RECORD_ROOT.INPUT]
        )
        self.assertEqual(
            "response to test",
            record_attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
        )
        # Verify second span.
        record_attributes = events["record_attributes"].iloc[1]
        self.assertEqual(
            SpanAttributes.SpanType.UNKNOWN,
            record_attributes[SpanAttributes.SPAN_TYPE],
        )
        self.assertEqual(
            7, record_attributes[SpanAttributes.CALL.KWARGS + ".n"]
        )
        self.assertEqual(49, record_attributes[SpanAttributes.CALL.RETURN])
        # Verify recording.
        self.assertEqual(1, len(recording))
        self.assertEqual(
            record_attributes[SpanAttributes.RECORD_ID],
            recording.get().record_id,
        )

    @pytest.mark.optional
    def test_legacy_tru_chain_app(self) -> None:
        responses = ["response to test"]
        llm = FakeListLLM(responses=responses)
        prompt = PromptTemplate(input_variables=["query"], template="{query}")
        app = LLMChain(llm=llm, prompt=prompt)
        tru_app = TruChain(app, app_name="MyTruChainApp", app_version="v1")
        with tru_app:
            app.run("test")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(4, len(events))
        # Verify first span.
        record_attributes = events["record_attributes"].iloc[0]
        self.assertEqual(
            SpanAttributes.SpanType.RECORD_ROOT,
            record_attributes[SpanAttributes.SPAN_TYPE],
        )
        self.assertEqual(
            "test", record_attributes[SpanAttributes.RECORD_ROOT.INPUT]
        )
        self.assertEqual(
            "response to test",
            record_attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
        )
        # Verify other spans.
        for i in range(1, events.shape[0]):
            record_attributes = events["record_attributes"].iloc[i]
            self.assertEqual(
                SpanAttributes.SpanType.UNKNOWN,
                record_attributes[SpanAttributes.SPAN_TYPE],
            )

    # TODO(otel): create a test like this for TruLlama.
    # @pytest.mark.optional
    # def test_legacy_tru_llama_app(self) -> None:
    #     llm = MockLLM()
    #     tru_app = TruLlama(llm, app_name="MyTruLlamaApp", app_version="v1")
    #     with tru_app:
    #         llm("test")
    #     TruSession().force_flush()
    #     events = self._get_events()
    #     self.assertEqual(2, len(events))
    #     # Verify first span.
    #     record_attributes = events["record_attributes"].iloc[0]
    #     self.assertEqual(
    #         SpanAttributes.SpanType.RECORD_ROOT,
    #         record_attributes[SpanAttributes.SPAN_TYPE],
    #     )
    #     self.assertEqual(
    #         "test", record_attributes[SpanAttributes.RECORD_ROOT.INPUT]
    #     )
    #     self.assertEqual(
    #         "response to test",
    #         record_attributes[SpanAttributes.RECORD_ROOT.OUTPUT],
    #     )
    #     # Verify second span.
    #     record_attributes = events["record_attributes"].iloc[1]
    #     self.assertEqual(
    #         SpanAttributes.SpanType.UNKNOWN,
    #         record_attributes[SpanAttributes.SPAN_TYPE],
    #     )
