"""Tests for the task pool every feedback run goes through."""

from concurrent import futures
import logging
import time
from unittest import TestCase
from unittest.mock import MagicMock

from trulens.core.schema import app as app_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.utils.threading import TP


class TestTP(TestCase):
    def test_submit_runs_the_task(self) -> None:
        self.assertEqual(TP().submit(lambda a, b: a + b, 1, b=2).result(10), 3)

    def test_submit_gives_up_after_the_timeout(self) -> None:
        future = TP().submit(lambda: time.sleep(30), timeout=0.1)

        with self.assertRaises(futures.TimeoutError):
            future.result(10)


class TestFeedbackScheduling(TestCase):
    """`_submit_feedback_functions` is how a recorded app schedules its
    feedback functions, so it must keep working with the pool above."""

    def test_feedback_functions_are_scheduled(self) -> None:
        app = app_schema.AppDefinition(
            app_name="scheduling_test",
            app_version="v1",
            root_class={"name": "App", "module": {"module_name": "app"}},
            app={},
        )
        record = record_schema.Record(app_id=app.app_id, calls=[])
        result = feedback_schema.FeedbackResult(
            feedback_result_id="fres", record_id=record.record_id, name="dummy"
        )
        feedback_function = MagicMock()
        feedback_function.name = "dummy"
        feedback_function.run.return_value = result

        scheduled = app_schema.AppDefinition._submit_feedback_functions(
            record=record,
            feedback_functions=[feedback_function],
            connector=MagicMock(),
            app=app,
        )

        self.assertEqual(
            [future.result(10) for _, future in scheduled], [result]
        )

    def test_failing_on_done_keeps_the_result_and_is_logged(self) -> None:
        app = app_schema.AppDefinition(
            app_name="scheduling_test",
            app_version="v1",
            root_class={"name": "App", "module": {"module_name": "app"}},
            app={},
        )
        record = record_schema.Record(app_id=app.app_id, calls=[])
        result = feedback_schema.FeedbackResult(
            feedback_result_id="fres", record_id=record.record_id, name="dummy"
        )
        feedback_function = MagicMock()
        feedback_function.name = "dummy"
        feedback_function.run.return_value = result

        def on_done(_: feedback_schema.FeedbackResult) -> None:
            raise ValueError("callback failed")

        with self.assertLogs(
            "trulens.core.schema.app", level=logging.ERROR
        ) as logs:
            scheduled = app_schema.AppDefinition._submit_feedback_functions(
                record=record,
                feedback_functions=[feedback_function],
                connector=MagicMock(),
                app=app,
                on_done=on_done,
            )
            self.assertEqual(
                [future.result(10) for _, future in scheduled], [result]
            )

        self.assertIn("callback failed", "\n".join(logs.output))
