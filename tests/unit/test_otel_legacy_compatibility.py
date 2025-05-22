import importlib
import os

import trulens.apps.app
from trulens.apps.app import TruApp
from trulens.apps.app import instrument as legacy_instrument
from trulens.core.otel.instrument import instrument as otel_instrument
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_test_case import OtelTestCase


class _LegacyTestApp:
    @legacy_instrument
    def respond_to_query(self, query: str) -> str:
        return f"response to {query}"


class TestOtelLegacyCompatibility(OtelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if is_otel_tracing_enabled():
            raise ValueError(
                "TRULENS_OTEL_TRACING must be disabled *initially* for these tests!"
            )
        os.environ["TRULENS_OTEL_TRACING"] = "1"
        super().setUpClass()

    def test_import(self) -> None:
        from trulens.apps.app import instrument

        self.assertIs(instrument, legacy_instrument)
        importlib.reload(trulens.apps.app)
        from trulens.apps.app import instrument

        self.assertIsInstance(instrument, otel_instrument)

    def test_legacy_app(self) -> None:
        app = _LegacyTestApp()
        tru_app = TruApp(
            app,
            main_method=app.respond_to_query,  # TODO(otel): This is not backwards compatible!
            app_name="MyCustomApp",
            app_version="v1",
        )
        with tru_app as recording:
            app.respond_to_query("test")
        TruSession().force_flush()
        events = self._get_events()
        self.assertEqual(1, len(events))
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
        self.assertEqual(1, len(recording))
        self.assertEqual(
            record_attributes[SpanAttributes.RECORD_ID], recording.get()
        )
