"""
Tests for OTEL TruBasic app.
"""

from unittest import main

from trulens.apps.basic import TruBasicApp
from trulens.core.session import TruSession

from tests.util.otel_app_test_case import OtelAppTestCase


class TestOtelTruBasic(OtelAppTestCase):
    def test_smoke(self) -> None:
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create and run app.
        basic_app = TruBasicApp(text_to_text=lambda name: f"Hi, {name}!")
        with basic_app(run_name="test run", input_id="42"):
            basic_app.app("Kojikun")
        # Compare results to expected.
        # TODO(otel): once we have `main_method` functionality from Daniel, we
        #             should have the `text_to_text` function be a MAIN span.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_basic__test_smoke.csv"
        )


if __name__ == "__main__":
    main()
