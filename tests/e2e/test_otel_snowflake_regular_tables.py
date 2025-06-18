import gc
import weakref

from trulens.apps.app import TruApp
from trulens.otel.semconv.trace import SpanAttributes

import tests.unit.test_otel_tru_custom
from tests.util.otel_test_case import OtelTestCase
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestOtelSnowflakeRegularTables(OtelTestCase, SnowflakeTestCase):
    def test_smoke(self):
        # Create session.
        self.get_session(
            app_base_name="test_otel_snowflake_regular_tables__test_smoke",
            init_server_side=False,
            init_server_side_with_staged_packages=False,
            use_account_event_table=False,
        )
        # Create app.
        app = tests.unit.test_otel_tru_custom.TestApp()
        tru_app = TruApp(app, app_name="default_app", app_version="base")
        with tru_app:
            app.respond_to_query("test")
        with tru_app:
            app.respond_to_query("throw")
        # Compare results to expected.
        ignore_locators = [
            f"[record_attributes][{SpanAttributes.RUN_NAME}]",
            f"[record_attributes][{SpanAttributes.INPUT_ID}]",
        ]
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_custom__test_smoke.csv",
            ignore_locators=ignore_locators,
        )
        # Check we can still call the app after recording once.
        with tru_app:
            app.respond_to_query("throw")
        # Check garbage collection.
        custom_app_ref = weakref.ref(tru_app)
        del tru_app
        gc.collect()
        self.assertCollected(custom_app_ref)
