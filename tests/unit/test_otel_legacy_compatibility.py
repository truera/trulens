import importlib
import os

import trulens.apps.app
from trulens.apps.app import instrument as legacy_instrument
from trulens.core.otel.instrument import instrument as otel_instrument
from trulens.core.otel.utils import is_otel_tracing_enabled

from tests.util.otel_test_case import OtelTestCase


class TestOtelLegacyCompatibility(OtelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if is_otel_tracing_enabled():
            raise ValueError(
                "TRULENS_OTEL_TRACING must be disabled *initially* for these tests!"
            )
        os.environ["TRULENS_OTEL_TRACING"] = "1"
        super().setUpClass()

    def test_legacy_instrument(self) -> None:
        from trulens.apps.app import instrument

        self.assertIs(instrument, legacy_instrument)
        importlib.reload(trulens.apps.app)
        from trulens.apps.app import instrument

        self.assertIsInstance(instrument, otel_instrument)
