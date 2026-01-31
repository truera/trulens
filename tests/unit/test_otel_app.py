from trulens.apps.app import TruApp
from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument

from tests.util.otel_test_case import OtelTestCase


class TestOtelApp(OtelTestCase):
    def test_invalid_selectors(self) -> None:
        # Create feedback function.
        def custom(
            name: str, greeting: str, optional_name: str = "Kojikun"
        ) -> float:
            return 1.0

        f_custom = Feedback(custom, name="custom").on({
            "name": Selector.select_record_input()
        })

        # Create app.
        class Greeter:
            @instrument()
            def greet(self, name: str) -> str:
                return f"Hello, {name}!"

        app = Greeter()
        with self.assertRaisesRegex(
            ValueError,
            "^Metric function `custom` has missing selectors:\n"
            "Missing selectors: \['greeting'\]\n"
            "Required function args: \['name', 'greeting'\]\n$",
        ):
            TruApp(
                app, app_name="greeter", app_version="v1", feedbacks=[f_custom]
            )
