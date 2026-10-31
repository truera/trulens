"""Tests for the debug log messages emitted while registering instrumented methods."""

import logging
from unittest import mock

from trulens.core import instruments as core_instruments

from tests import test as mod_test


class Component:
    def run(self):
        pass


class TestInstrumentDebugLogging(mod_test.TruTestCase):
    def setUp(self):
        super().setUp()
        self.logger = logging.getLogger(core_instruments.__name__)

    def test_decorating_message(self):
        with self.assertLogs(self.logger, level="DEBUG") as logs:
            core_instruments.instrument(Component.run)

        self.assertEqual(
            [record.getMessage() for record in logs.records],
            [f"decorating {Component.run}"],
        )

    def test_adding_method_message(self):
        default = core_instruments.Instrument.Default
        with (
            mock.patch.object(default, "MODULES", set()),
            mock.patch.object(default, "CLASSES", set()),
            mock.patch.object(default, "METHODS", []),
            self.assertLogs(self.logger, level="DEBUG") as logs,
        ):
            core_instruments.instrument.method(Component, "run")

        self.assertEqual(
            [record.getMessage() for record in logs.records],
            [f"adding method run of {Component} from module {__name__}"],
        )
