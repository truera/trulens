"""
Tests for OTEL TruBasic app.
"""

from trulens.apps.basic import TruBasicApp

import tests.util.otel_tru_app_test_case


class TestOtelTruBasic(tests.util.otel_tru_app_test_case.OtelTruAppTestCase):
    @staticmethod
    def _create_test_app_info() -> (
        tests.util.otel_tru_app_test_case.TestAppInfo
    ):
        text_to_text_fn = lambda name: f"Hi, {name}!"
        return tests.util.otel_tru_app_test_case.TestAppInfo(
            app=text_to_text_fn,
            main_method=None,
            TruAppClass=TruBasicApp,
        )

    def test_smoke(self) -> None:
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
