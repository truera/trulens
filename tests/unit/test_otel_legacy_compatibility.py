import importlib
import os

import trulens.apps.app
from trulens.apps.app import TruApp
from trulens.apps.app import legacy_instrument
from trulens.core.otel.instrument import instrument as otel_instrument
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase

try:
    # These imports require optional dependencies to be installed.
    from langchain.chains import LLMChain
    from langchain.llms.fake import FakeListLLM
    from langchain.prompts import PromptTemplate
    from trulens.apps.langchain import TruChain
except Exception:
    pass


class TestOtelLegacyCompatibility(OtelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if is_otel_tracing_enabled():
            raise ValueError(
                "TRULENS_OTEL_TRACING must be disabled *initially* for these tests!"
            )
        os.environ["TRULENS_OTEL_BACKWARDS_COMPATIBILITY"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(self) -> None:
        del os.environ["TRULENS_OTEL_BACKWARDS_COMPATIBILITY"]
        super().tearDownClass()

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

    def test_legacy_custom_app(self) -> None:
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
        tru_app = TruApp(app, app_name="MyApp", app_version="v1")
        with tru_app as recording:
            app.respond_to_query("test")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(2, len(events))
        self.assertIsNone(recording)
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

    def test_legacy_tru_chain_app(self) -> None:
        responses = ["response to test"]
        llm = FakeListLLM(responses=responses)
        prompt = PromptTemplate(input_variables=["query"], template="{query}")
        app = LLMChain(llm=llm, prompt=prompt)
        tru_app = TruChain(app, app_name="MyTruChain", app_version="v1")
        with tru_app as recording:
            app.run("test")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(4, len(events))
        self.assertIsNone(recording)
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
