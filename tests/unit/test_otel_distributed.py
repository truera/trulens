from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import multiprocessing
import os
import time

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
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        full_scoped_attributes=lambda ret, exception, *args, **kwargs: {
            "process_id": os.getpid()
        },
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
            capitalized_name = self.capitalize(name)
            TruSession().force_flush()
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(capitalized_name.encode())
        elif self.path.startswith("/ping"):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
        else:
            raise ValueError("Unknown path!")

    @instrument(
        full_scoped_attributes=lambda ret, exception, *args, **kwargs: {
            "process_id": os.getpid()
        }
    )
    def capitalize(self, name: str) -> str:
        return name.upper()


def run_server():
    os.environ["TRULENS_OTEL_TRACING"] = "1"
    TruSession()  # This starts an exporter.
    server = HTTPServer(("localhost", 8000), CapitalizeHandler)
    server.serve_forever()


class TestOtelDistributed(OtelAppTestCase):
    @staticmethod
    def _wait_for_server(
        num_retries: int = 40, sleep_time: float = 0.25
    ) -> None:
        server_up = False
        for _ in range(num_retries):
            try:
                res = requests.get("http://localhost:8000/ping")
                if res.status_code == 200:
                    server_up = True
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(sleep_time)
        if not server_up:
            raise ValueError("Server not up.")

    def setUp(self) -> None:
        super().setUp()
        self.server_process = multiprocessing.Process(target=run_server)
        self.server_process.start()
        self._wait_for_server()

    def tearDown(self) -> None:
        self.server_process.terminate()
        self.server_process.join()
        super().tearDown()

    def test_distributed(self) -> None:
        # Create TruApp that makes a network call.
        test_app = _TestApp()
        custom_app = TruApp(test_app, main_method=test_app.greet)
        recorder = custom_app(run_name="test run", input_id="789")
        with recorder:
            test_app.greet("test")
        # Compare results to expected.
        TruSession().force_flush()
        actual = self._get_events()
        self.assertEqual(len(actual), 3)
        self.assertNotEqual(
            actual.iloc[1]["record_attributes"]["process_id"],
            actual.iloc[2]["record_attributes"]["process_id"],
        )
        for attribute in [
            SpanAttributes.APP_NAME,
            SpanAttributes.APP_VERSION,
            SpanAttributes.RECORD_ID,
            SpanAttributes.RUN_NAME,
            SpanAttributes.INPUT_ID,
        ]:
            self.assertEqual(
                actual.iloc[1]["record_attributes"][attribute],
                actual.iloc[2]["record_attributes"][attribute],
            )
