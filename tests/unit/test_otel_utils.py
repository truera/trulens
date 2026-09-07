"""Tests for OTEL enablement helpers in `trulens.core.otel.utils`."""

import logging
import os
from unittest import TestCase
from unittest import mock

from trulens.core.otel import utils as otel_utils


class TestOtelUtils(TestCase):
    def setUp(self) -> None:
        # The disabled-warning is emitted once per process; reset between tests.
        otel_utils._OTEL_DISABLED_WARNING_EMITTED = False

    def test_tracing_enabled_by_default(self) -> None:
        """No environment variable means OTEL tracing is on."""
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(otel_utils.is_otel_tracing_enabled())

    def test_setting_to_1_is_a_noop(self) -> None:
        """`TRULENS_OTEL_TRACING=1` is redundant, not required."""
        for value in ["1", "true", "True", "yes", "anything"]:
            with mock.patch.dict(
                os.environ, {"TRULENS_OTEL_TRACING": value}, clear=True
            ):
                self.assertTrue(
                    otel_utils.is_otel_tracing_enabled(),
                    f"expected enabled for TRULENS_OTEL_TRACING={value!r}",
                )

    def test_only_0_and_false_disable(self) -> None:
        for value in ["0", "false", "FALSE", "False"]:
            otel_utils._OTEL_DISABLED_WARNING_EMITTED = False
            with mock.patch.dict(
                os.environ, {"TRULENS_OTEL_TRACING": value}, clear=True
            ):
                self.assertFalse(
                    otel_utils.is_otel_tracing_enabled(),
                    f"expected disabled for TRULENS_OTEL_TRACING={value!r}",
                )

    def test_warns_when_disabled(self) -> None:
        """Disabling tracing is otherwise silent, so it must warn."""
        with mock.patch.dict(
            os.environ, {"TRULENS_OTEL_TRACING": "0"}, clear=True
        ):
            with self.assertLogs(
                otel_utils.logger, level=logging.WARNING
            ) as cm:
                otel_utils.is_otel_tracing_enabled()
        self.assertTrue(
            any("TRULENS_OTEL_TRACING" in line for line in cm.output),
            f"warning should name the variable, got: {cm.output}",
        )

    def test_warns_only_once(self) -> None:
        """Called from many hot paths; must not spam the log."""
        with mock.patch.dict(
            os.environ, {"TRULENS_OTEL_TRACING": "0"}, clear=True
        ):
            with mock.patch.object(otel_utils.logger, "warning") as warn:
                for _ in range(5):
                    otel_utils.is_otel_tracing_enabled()
        self.assertEqual(warn.call_count, 1)

    def test_no_warning_when_enabled(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(otel_utils.logger, "warning") as warn:
                otel_utils.is_otel_tracing_enabled()
        warn.assert_not_called()

    def test_backwards_compatibility_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                otel_utils.is_otel_backwards_compatibility_enabled()
            )
        with mock.patch.dict(
            os.environ,
            {"TRULENS_OTEL_BACKWARDS_COMPATIBILITY": "0"},
            clear=True,
        ):
            self.assertFalse(
                otel_utils.is_otel_backwards_compatibility_enabled()
            )
