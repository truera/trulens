"""
Tests for OTEL TruBasic app.
"""

from trulens.apps.basic import TruBasicApp
from trulens.core.session import TruSession

from tests.util.otel_app_test_case import OtelAppTestCase


class TestOtelTruBasic(OtelAppTestCase):
    def test_smoke(self) -> None:
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create and run app.
        text_to_text_fn = lambda name: f"Hi, {name}!"
        basic_app = TruBasicApp(text_to_text=text_to_text_fn)
        basic_app.instrumented_invoke_main_method(
            run_name="test run",
            input_id="42",
            main_method_args=("Kojikun",),
        )
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_basic__test_smoke.csv"
        )
