from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import multiprocessing
import os
import time
import unittest

import opentelemetry.context as context_api
from opentelemetry.propagate import extract
from opentelemetry.propagate import inject
import requests
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase


class _TestApp:
    @instrument(
        span_type=SpanAttributes.SpanType.MAIN,
        full_scoped_attributes={"process_id": os.getpid()},
    )
    def greet(self, name: str) -> str:
        headers = {}
        inject(headers, context=context_api.get_current())
        response = requests.get(
            f"http://localhost:8000/capitalize?name={name}", headers=headers
        )
        return response.text


class CapitalizeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/capitalize"):
            context = extract(self.headers)
            context_api.attach(context)
            name = self.path.split("=")[1]
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(self.capitalize(name).encode())
            TruSession().force_flush()
        elif self.path.startswith("/ping"):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
        else:
            raise ValueError("Unknown path!")

    @instrument(full_scoped_attributes={"process_id": os.getpid()})
    def capitalize(self, name: str) -> str:
        return name.upper()


def run_server():
    os.environ["TRULENS_OTEL_TRACING"] = "1"
    TruSession()  # This starts an exporter.
    server = HTTPServer(("localhost", 8000), CapitalizeHandler)
    server.serve_forever()


class TestOtelDistributed(OtelAppTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_process = multiprocessing.Process(target=run_server)
        cls.server_process.start()
        for _ in range(40):
            try:
                requests.get("http://localhost:8000/ping")
                break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.25)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.server_process.terminate()
        cls.server_process.join()
        super().tearDownClass()

    def test_distributed(self):
        # Set up.
        tru_session = TruSession()
        tru_session.reset_database()
        # Create TruApp that makes a network call.
        test_app = _TestApp()
        custom_app = TruApp(test_app, main_method=test_app.greet)
        recorder = custom_app(run_name="test run", input_id="789")
        with recorder:
            test_app.greet("test")
        # Compare results to expected.
        tru_session.force_flush()
        actual = self._get_events()
        self.assertEqual(len(actual), 3)
        self.assertNotEqual(
            actual.iloc[1]["record_attributes"]["process_id"],
            actual.iloc[2]["record_attributes"]["process_id"],
        )
        for attribute in [
            "app_name",
            "app_version",
            "record_id",
            "run_name",
            "input_id",
        ]:
            self.assertEqual(
                actual.iloc[1]["record_attributes"][
                    f"ai.observability.{attribute}"
                ],
                actual.iloc[2]["record_attributes"][
                    f"ai.observability.{attribute}"
                ],
            )


if __name__ == "__main__":
    unittest.main()
