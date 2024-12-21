"""
Tests for OTEL instrument decorator.
"""

from unittest import TestCase
from unittest import main

from trulens.apps.custom import TruCustomApp
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.init import init
from trulens.experimental.otel_tracing.core.instrument import instrument


class _TestApp:
    @instrument()
    def respond_to_query(self, query: str) -> str:
        return f"answer: {self.nested(query)}"

    @instrument(attributes={"nested_attr1": "value1"})
    def nested(self, query: str) -> str:
        return f"nested: {self.nested2(query)}"

    @instrument(
        attributes=lambda ret, exception, *args, **kwargs: {
            "nested2_ret": ret,
            "nested2_args[0]": args[0],
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
        if query == "throw":
            raise ValueError("nested3 exception")
        return "nested3"


class TestOtelInstrument(TestCase):
    def setUp(self):
        pass

    def test_deterministic_app_id(self):
        session = TruSession()
        session.experimental_enable_feature("otel_tracing")
        session.reset_database()
        init(session, debug=True)

        test_app = _TestApp()
        custom_app = TruCustomApp(test_app)

        with custom_app as recording:
            test_app.respond_to_query("test")

        with custom_app as recording:
            test_app.respond_to_query("throw")

        print(recording)


if __name__ == "__main__":
    main()
