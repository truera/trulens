"""Regression test for Run.compute_metrics message building.

compute_metrics accepts metrics as strings or as Metric/MetricConfig objects.
When an already-computed metric was passed as an object, the skip-path message
did ", ".join(computed_metrics) over the raw objects, which raised TypeError
instead of returning the user-facing "already computed" message.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

try:
    from trulens.core import Selector
    from trulens.core.dao.run import RunDaoBase
    from trulens.core.metric.metric import Metric
    from trulens.core.run import Run
except Exception:  # pragma: no cover
    Run = None


def _make_run() -> "Run":
    return Run.model_validate({
        "run_name": "test_run",
        "object_name": "TEST_AGENT",
        "object_type": "EXTERNAL AGENT",
        "object_version": "v1",
        "run_metadata": {},
        "source_info": {
            "name": "dummy_source",
            "column_spec": {"input": "INPUT"},
            "source_type": "TABLE",
        },
        "app": MagicMock(),
        "main_method_name": "dummy_method",
        "run_dao": MagicMock(spec=RunDaoBase),
        "tru_session": MagicMock(),
    })


def _make_metric(name: str) -> "Metric":
    def impl(output: str) -> float:
        return 1.0

    return Metric(
        implementation=impl,
        name=name,
        selectors={"output": Selector.select_record_output()},
    )


class TestComputeMetricsMessage(unittest.TestCase):
    def setUp(self):
        if Run is None:
            self.skipTest("Run not available.")
        self.run = _make_run()
        self.metadata = {
            "run_metadata": {
                "metrics": {
                    "m1": {
                        "name": "my_metric",
                        "completion_status": {
                            "status": Run.CompletionStatusStatus.COMPLETED
                        },
                    }
                }
            }
        }

    def test_already_computed_metric_object_returns_message(self):
        """Metric object passed to compute_metrics yields message when already
        computed (no TypeError)."""
        metric = _make_metric("my_metric")
        with (
            patch.object(Run, "describe", return_value=self.metadata),
            patch.object(
                Run, "get_status", return_value="INVOCATION_COMPLETED"
            ),
            patch.object(
                Run, "_can_start_new_metric_computation", return_value=True
            ),
        ):
            result = self.run.compute_metrics([metric])
        self.assertIsInstance(result, str)
        self.assertIn("already computed", result)
        self.assertIn("my_metric", result)

    def test_already_computed_metric_string_still_works(self):
        with (
            patch.object(Run, "describe", return_value=self.metadata),
            patch.object(
                Run, "get_status", return_value="INVOCATION_COMPLETED"
            ),
            patch.object(
                Run, "_can_start_new_metric_computation", return_value=True
            ),
        ):
            result = self.run.compute_metrics(["my_metric"])
        self.assertIsInstance(result, str)
        self.assertIn("my_metric", result)


if __name__ == "__main__":
    unittest.main()
