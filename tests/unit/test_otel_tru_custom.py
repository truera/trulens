"""
Tests for OTEL instrument decorator and custom app.
"""

import traceback
from unittest import main

from trulens.apps.custom import TruCustomApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase

_query_to_nested3_trace = {}


class TestApp:
    @instrument(span_type=SpanAttributes.SpanType.MAIN)
    def respond_to_query(self, query: str) -> str:
        return f"answer: {self.nested(query)}"

    @instrument(attributes={"nested_attr1": "value1"})
    def nested(self, query: str) -> str:
        return f"nested: {self.nested2(query)}"

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            "nested2_ret": ret,
            "nested2_args[1]": args[1],
        }
    )
    def nested2(self, query: str) -> str:
        nested_result = ""

        try:
            nested_result = self.nested3(query)
        except Exception:
            pass

        return f"nested2: {nested_result}"

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            "nested3_ex": exception.args if exception else None,
            "nested3_ret": ret,
            "selector_name": "special",
            "cows": "moo",
        }
    )
    def nested3(self, query: str) -> str:
        _query_to_nested3_trace[query] = "".join(traceback.format_stack())
        if query == "throw":
            raise ValueError("nested3 exception")
        return "nested3"


class TestOtelTruCustom(OtelAppTestCase):
    def test_smoke(self) -> None:
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create and run app.
        test_app = TestApp()
        custom_app = TruCustomApp(test_app)
        with custom_app(run_name="test run", input_id="456"):
            test_app.respond_to_query("test")
        with custom_app():
            test_app.respond_to_query("throw")
        # Verify stack trace of nested3.
        tru_session.experimental_force_flush()
        events = self._get_events()
        self.assertEqual(len(events), 10)
        self.assertEqual(
            events.iloc[4]["record_attributes"][SpanAttributes.CALL.STACK],
            _query_to_nested3_trace["test"],
        )
        self.assertEqual(
            events.iloc[9]["record_attributes"][SpanAttributes.CALL.STACK],
            _query_to_nested3_trace["throw"],
        )
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_custom__test_smoke.csv"
        )


if __name__ == "__main__":
    main()
