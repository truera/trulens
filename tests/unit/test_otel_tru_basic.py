"""
Tests for OTEL TruBasic app.
"""

from trulens.apps.basic import TruBasicApp

from tests.util.otel_app_test_case import OtelAppTestCase


class TestOtelTruBasic(OtelAppTestCase):
    def test_smoke(self) -> None:
        # Create and run app.
        text_to_text_fn = lambda name: f"Hi, {name}!"
        basic_app = TruBasicApp(
            text_to_text=text_to_text_fn, main_method=text_to_text_fn
        )
        with basic_app(run_name="test run", input_id="42"):
            basic_app.app("Kojikun")
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_basic__test_smoke.csv"
        )
