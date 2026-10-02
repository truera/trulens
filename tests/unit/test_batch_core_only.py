"""Batch evaluation must work without the optional feedback package."""

import subprocess
import sys
import textwrap


def test_batch_evaluation_without_feedback_package():
    """Evaluate custom metrics and sentinels with feedback imports disabled."""
    script = textwrap.dedent(
        """
        import sys

        # Simulate a core-only installation even in the optional-deps CI job.
        sys.modules["trulens.feedback"] = None

        from trulens.core import batch as core_batch
        from trulens.core.feedback import selector as feedback_selector
        from trulens.core.metric import metric as core_metric

        def identity(value):
            return value

        metric = core_metric.Metric(
            name="score",
            implementation=identity,
            selectors={
                "value": feedback_selector.Selector.from_column(
                    "values", collect_list=False
                )
            },
        )
        result = core_batch.BatchEvaluator(
            [metric], max_workers=1
        ).evaluate([
            {"values": [0.5]},
            {"values": [0.0, 1.0]},
            {"values": [1.0, -1.0]},
            {"values": [-1.0]},
        ])
        assert result["score"].tolist() == [0.5, 0.5, 1.0, -1.0]
        assert "unparsable" in result.iloc[3]["score_explanation"]["error"]
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
