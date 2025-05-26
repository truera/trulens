"""
Tests for OTEL instrument decorator and custom app.
"""

import gc
import weakref

from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

import tests.util.otel_tru_app_test_case
from tests.utils import enable_otel_backwards_compatibility


class TestApp:
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={"the_query": "query", "the_return": "return"},
    )
    def respond_to_query(self, query: str) -> str:
        return f"answer: {self.nested(query)}"

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
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


class TestOtelTruCustom(tests.util.otel_tru_app_test_case.OtelTruAppTestCase):
    @staticmethod
    def _create_test_app_info() -> (
        tests.util.otel_tru_app_test_case.TestAppInfo
    ):
        app = TestApp()
        return tests.util.otel_tru_app_test_case.TestAppInfo(
            app=app, main_method=app.respond_to_query, TruAppClass=TruApp
        )

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
        # Check we can still call the app after recording once.
        with custom_app:
            test_app.respond_to_query("throw")
        # Check garbage collection.
        custom_app_ref = weakref.ref(custom_app)
        del custom_app
        gc.collect()
        self.assertCollected(custom_app_ref)

    @enable_otel_backwards_compatibility
    def test_legacy_app(self) -> None:
        # Create and run app.
        test_app = TestApp()
        custom_app = TruApp(test_app)
        with custom_app:
            test_app.respond_to_query("test")
        with custom_app:
            test_app.respond_to_query("throw")
        # Compare results to expected.
        self._compare_record_attributes_to_golden_dataframe(
            "tests/unit/static/golden/test_otel_tru_custom__test_smoke.csv"
        )

    def test_incorrect_span_attributes(self) -> None:
        class MyProblematicApp:
            @instrument(
                attributes=lambda ret, exception, *args, **kwargs: kwargs[
                    "does_not_exist"
                ]
            )
            def say_hi(self):
                return "Hi!"

        app = MyProblematicApp()
        tru_app = TruApp(
            app,
            app_name="MyProblematicApp",
            app_version="v1",
            main_method=app.say_hi,
        )
        with tru_app:
            with self.assertRaisesRegex(KeyError, "does_not_exist"):
                app.say_hi()
