import unittest

from trulens.apps.app import TruApp

from tests.util.otel_app_test_case import OtelAppTestCase


# Dummy application to use with TruApp.
class DummyApp:
    def respond_to_query(self, query: str) -> str:
        return "dummy response"


class TestOtelMainMethod(OtelAppTestCase):
    def test_missing_main_method_raises_error(self):
        dummy_app = DummyApp()
        # Attempt to create a TruApp without specifying main_method.
        with self.assertRaises(ValueError) as context:
            TruApp(
                dummy_app,
                app_name="Dummy App",
                app_version="v1",
                # Intentionally omitting main_method
            )
        self.assertIn("main_method", str(context.exception))


if __name__ == "__main__":
    unittest.main()
