"""
Tests for OTEL instrument decorator and custom app.
"""

from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase


class TestApp:
    @instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT)
    def respond_to_query(self, query: str) -> str:
        return f"answer: {self.nested(query)}"

    @instrument(
        attributes=lambda ret, exception, *args, **kargs: {
            f"{SpanAttributes.UNKNOWN.base}.nested_attr1": "value1"
        }
    )
    def nested(self, query: str) -> str:
        return f"nested: {self.nested2(query)}"

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            f"{SpanAttributes.UNKNOWN.base}.nested2_ret": ret,
            f"{SpanAttributes.UNKNOWN.base}.nested2_args[1]": args[1],
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
            f"{SpanAttributes.UNKNOWN.base}.nested3_ex": exception.args
            if exception
            else None,
            f"{SpanAttributes.UNKNOWN.base}.nested3_ret": ret,
            f"{SpanAttributes.UNKNOWN.base}.selector_name": "special",
            f"{SpanAttributes.UNKNOWN.base}.cows": "moo",
        }
    )
    def nested3(self, query: str) -> str:
        if query == "throw":
            raise ValueError("nested3 exception")
        return "nested3"


class TestOtelTruCustom(OtelAppTestCase):
    def test_smoke(self) -> None:
        # Create and run app.
        test_app = TestApp()
        custom_app = TruApp(test_app, main_method=test_app.respond_to_query)
        custom_app.instrumented_invoke_main_method(
            "test run", "42", main_method_args=("test",)
        )
        with custom_app:
            test_app.respond_to_query("throw")
        # Compare results to expected.
        self._compare_events_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_custom__test_smoke.csv"
        )
