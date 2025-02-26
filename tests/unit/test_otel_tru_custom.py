"""
Tests for OTEL instrument decorator and custom app.
"""

from trulens.apps.app import TruApp
from trulens.core.database.connector import DefaultDBConnector
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

import tests.util.otel_tru_app_test_case


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

    def test_with_existing_tru_session(self) -> None:
        self.clear_TruSession_singleton()
        connector1 = DefaultDBConnector()
        connector2 = DefaultDBConnector()
        self.assertIsNot(connector1, connector2)
        TruSession(connector=connector1)
        test_app = TestApp()
        with self.assertRaisesRegex(
            ValueError,
            "Already created `TruSession` with different `connector`!",
        ):
            TruApp(
                test_app,
                main_method=test_app.respond_to_query,
                connector=connector2,
            )
        tru_app = TruApp(
            test_app,
            main_method=test_app.respond_to_query,
            connector=connector1,
        )
        self.assertIs(connector1, tru_app.connector)

    def test_with_new_tru_session(self) -> None:
        self.clear_TruSession_singleton()
        connector = DefaultDBConnector()
        test_app = TestApp()
        tru_app = TruApp(
            test_app,
            main_method=test_app.respond_to_query,
            connector=connector,
        )
        self.assertIs(connector, tru_app.connector)
        self.assertIs(connector, TruSession().connector)
