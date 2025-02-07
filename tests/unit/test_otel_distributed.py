from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import logging
import logging.handlers
import multiprocessing
import os
import pprint
import time
import unittest

import opentelemetry.context as context_api
from opentelemetry.propagate import extract
from opentelemetry.propagate import inject
import pandas as pd
import requests
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.core.session import TruSession
from trulens.experimental.otel_tracing.core.exporter.connector import (
    set_up_logging,
)
from trulens.otel.semconv.trace import SpanAttributes

from tests.util.otel_app_test_case import OtelAppTestCase

LOG_FILE = "/tmp/all_logs.txt"


class _TestApp:
    @instrument(
        span_type=SpanAttributes.SpanType.MAIN,
        full_scoped_attributes={"process_id": os.getpid(), "from_child": False},
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
        full_scoped_attributes={"process_id": os.getpid(), "from_child": True}
    )
    def capitalize(self, name: str) -> str:
        return name.upper()


def run_server():
    set_up_logging(log_level=logging.DEBUG, start_fresh=False)
    logger = logging.getLogger(__name__)
    logger.info("THIS IS THE CHILD!")
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
        set_up_logging(log_level=logging.DEBUG)
        time.sleep(1)
        super().setUp()
        TruSession()
        time.sleep(1)
        logger = logging.getLogger(__name__)
        logger.info("THIS IS THE PARENT!")
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
            res = test_app.greet("test")
        # Compare results to expected.
        TruSession().force_flush()
        actual = self._get_events()
        for _ in range(10):
            print("START LOGS:")
        with open(LOG_FILE, "r") as fh:
            print(fh.read())
        for _ in range(10):
            print("STOP LOGS!")
        for _ in range(10):
            print("START EVENTS:")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        print(actual.T)
        print("JUST RECORD ATTRIBUTES:")
        for i, x in enumerate(actual["record_attributes"].to_numpy()):
            print(i)
            pprint.pprint(x)
        for _ in range(10):
            print("STOP EVENTS!")
        self.assertEqual(res, "TEST")
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
