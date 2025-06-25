import unittest
from unittest.mock import MagicMock

import pytest

try:
    from trulens.core.run import Run
    from trulens.core.run import RunStatus

except Exception:
    Run = None
    RunStatus = None


# Helper: create a dummy Run instance from a dict.
def create_dummy_run(run_metadata_dict: dict, run_status: str = None) -> Run:
    # Minimal required fields for Run model.
    base = {
        "run_name": "test_run",
        "object_name": "TEST_AGENT",
        "object_type": "EXTERNAL AGENT",
        "object_version": "v1",
        "run_metadata": run_metadata_dict,
        "source_info": {
            "name": "dummy_source",
            "column_spec": {"dummy": "dummy"},
            "source_type": "TABLE",
        },
    }
    # Extra fields needed by from_metadata_df.
    extra = {
        "app": MagicMock(),
        "main_method_name": "dummy_method",
        "run_dao": MagicMock(),
        "tru_session": MagicMock(),
    }

    run = Run.model_validate({**base, **extra})
    if run_status is not None:
        run.run_status = run_status
    return run


@pytest.mark.snowflake
class TestRunStatusOrchestration(unittest.TestCase):
    def setUp(self):
        if Run is None or RunStatus is None:
            self.skipTest("TruLens Run class not available.")
            return

        self.base_run_dao = MagicMock()
        self.base_run_dao.upsert_run_metadata_fields = MagicMock()
        self.mock_metrics = {}
        self.mock_run_metadata = {"metrics": self.mock_metrics}
        self.mock_describe = MagicMock(
            return_value={"run_metadata": self.mock_run_metadata}
        )

    def attach_run_dao(self, run: Run):
        run.run_dao = self.base_run_dao
        run.run_name = "test_run"
        run.object_name = "TEST_AGENT"
        run.object_type = "EXTERNAL AGENT"
        run.object_version = "v1"

    def create_invocation(
        self,
        id: str,
        start_time_ms: int,
        end_time_ms: int,
        completion_status: str = None,
    ):
        """Helper to build an invocation dict."""
        inv = {
            "id": id,
            "input_records_count": 1000,
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
        }
        if completion_status is not None:
            inv["completion_status"] = {
                "status": completion_status,
                "record_count": 1000,
            }
        else:
            inv["completion_status"] = None
        return inv

    def test_with_completion_status_completed(self):
        # Latest invocation has completion_status COMPLETED.
        inv = self.create_invocation(
            "inv1", 1000, 0, Run.CompletionStatusStatus.COMPLETED
        )
        run_metadata = {
            "invocations": {"inv1": inv},
            "metrics": {},
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)

        status = run._compute_latest_invocation_status(run)
        self.assertEqual(status, RunStatus.INVOCATION_COMPLETED)

    def test_with_completion_status_partially_completed(self):
        # Latest invocation has completion_status PARTIALLY_COMPLETED.
        inv = self.create_invocation(
            "inv1", 1000, 0, Run.CompletionStatusStatus.PARTIALLY_COMPLETED
        )
        run_metadata = {
            "invocations": {"inv1": inv},
            "metrics": {},
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)

        status = run._compute_latest_invocation_status(run)
        self.assertEqual(status, RunStatus.INVOCATION_PARTIALLY_COMPLETED)

    def test_with_completion_status_failed(self):
        # Latest invocation has completion_status FAILED.
        inv = self.create_invocation(
            "inv1", 1000, 0, Run.CompletionStatusStatus.FAILED
        )
        run_metadata = {
            "invocations": {"inv1": inv},
            "metrics": {},
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)

        status = run._compute_latest_invocation_status(run)
        self.assertEqual(status, RunStatus.FAILED)

    def test_all_metrics_completed(self):
        run_metadata = {
            "invocations": {
                "inv1": {
                    "id": "inv1",
                    "input_records_count": 1000,
                    "start_time_ms": 1000,
                    "end_time_ms": 0,
                    "completion_status": {
                        "status": "COMPLETED",
                        "record_count": 1000,
                    },
                }
            },
            "metrics": {
                "met1": {
                    "id": "met1",
                    "name": "answer_relevance",
                    "completion_status": {
                        "status": "COMPLETED",
                        "record_count": 10,
                    },
                    "computation_id": "comp1",
                }
            },
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)
        status = run._compute_overall_computations_status(run)
        self.assertEqual(status, RunStatus.COMPLETED)

    def test_all_metrics_failed(self):
        run_metadata = {
            "invocations": {
                "inv1": {
                    "id": "inv1",
                    "input_records_count": 1000,
                    "start_time_ms": 1000,
                    "end_time_ms": 0,
                    "completion_status": {
                        "status": "COMPLETED",
                        "record_count": 1000,
                    },
                }
            },
            "metrics": {
                "met1": {
                    "id": "met1",
                    "name": "answer_relevance",
                    "completion_status": {
                        "status": "FAILED",
                        "record_count": 5,
                    },
                    "computation_id": "comp1",
                },
                "met2": {
                    "id": "met2",
                    "name": "context_relevance",
                    "completion_status": {
                        "status": "FAILED",
                        "record_count": 5,
                    },
                    "computation_id": "comp1",
                },
            },
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)
        status = run._compute_overall_computations_status(run)
        self.assertEqual(status, RunStatus.FAILED)

    def test_metrics_mix_completed_failed(self):
        run_metadata = {
            "invocations": {
                "inv1": {
                    "id": "inv1",
                    "input_records_count": 1000,
                    "start_time_ms": 1000,
                    "end_time_ms": 0,
                    "completion_status": {
                        "status": "FAILED",
                        "record_count": 1000,
                    },
                }
            },
            "metrics": {
                "met1": {
                    "id": "met1",
                    "name": "answer_relevance",
                    "completion_status": {
                        "status": "COMPLETED",
                        "record_count": 10,
                    },
                    "computation_id": "comp1",
                },
                "met2": {
                    "id": "met2",
                    "name": "coherence",
                    "completion_status": {
                        "status": "FAILED",
                        "record_count": 7,
                    },
                    "computation_id": "comp1",
                },
            },
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)
        status = run._compute_overall_computations_status(run)
        self.assertEqual(status, RunStatus.PARTIALLY_COMPLETED)

    def create_real_run(self):
        return Run.model_validate({
            "run_name": "test_run",
            "object_name": "TEST_AGENT",
            "object_type": "EXTERNAL AGENT",
            "object_version": "v1",
            "run_metadata": {},
            "source_info": {
                "name": "dummy_source",
                "column_spec": {"dummy": "dummy"},
                "source_type": "TABLE",
            },
            "app": MagicMock(),
            "main_method_name": "dummy_method",
            "run_dao": MagicMock(),
            "tru_session": MagicMock(),
        })

    def test_metrics_none(self):
        run = self.create_real_run()
        self.mock_run_metadata["metrics"] = None
        run.describe = self.mock_describe

        result = run._should_skip_computation("answer_relevance", run)
        self.assertFalse(result)

    def test_no_matching_metric(self):
        run = self.create_real_run()
        run.describe = self.mock_describe

        self.mock_metrics["met1"] = {
            "name": "other_metric",
            "completion_status": {
                "status": Run.CompletionStatusStatus.COMPLETED,
                "record_count": 10,
            },
        }

        result = run._should_skip_computation("answer_relevance", run)
        self.assertFalse(result)

    def test_one_completed_metric(self):
        run = self.create_real_run()
        run.describe = self.mock_describe

        self.mock_metrics["met1"] = {
            "name": "answer_relevance",
            "completion_status": {
                "status": Run.CompletionStatusStatus.COMPLETED,
                "record_count": 10,
            },
        }

        result = run._should_skip_computation("answer_relevance", run)
        self.assertTrue(result)

    def test_one_in_progress_metric(self):
        run = self.create_real_run()
        run.describe = self.mock_describe

        self.mock_metrics["met1"] = {
            "name": "answer_relevance",
            "completion_status": None,
        }

        result = run._should_skip_computation("answer_relevance", run)
        self.assertTrue(result)

    def test_all_failed_metric(self):
        run = self.create_real_run()
        run.describe = self.mock_describe

        self.mock_metrics["met1"] = {
            "name": "answer_relevance",
            "completion_status": {
                "status": Run.CompletionStatusStatus.FAILED,
                "record_count": 5,
            },
        }

        result = run._should_skip_computation("answer_relevance", run)
        self.assertFalse(result)

    def test_multiple_metrics_one_in_progress(self):
        run = self.create_real_run()
        run.describe = self.mock_describe

        self.mock_metrics["met1"] = {
            "name": "answer_relevance",
            "completion_status": {
                "status": Run.CompletionStatusStatus.FAILED,
                "record_count": 5,
            },
        }
        self.mock_metrics["met2"] = {
            "name": "answer_relevance",
            "completion_status": None,
        }

        result = run._should_skip_computation("answer_relevance", run)
        self.assertTrue(result)

    def test_multiple_metrics_one_completed(self):
        run = self.create_real_run()
        run.describe = self.mock_describe

        self.mock_metrics["met1"] = {
            "name": "answer_relevance",
            "completion_status": {
                "status": Run.CompletionStatusStatus.FAILED,
                "record_count": 5,
            },
        }
        self.mock_metrics["met2"] = {
            "name": "answer_relevance",
            "completion_status": {
                "status": Run.CompletionStatusStatus.COMPLETED,
                "record_count": 10,
            },
        }

        result = run._should_skip_computation("answer_relevance", run)
        self.assertTrue(result)
