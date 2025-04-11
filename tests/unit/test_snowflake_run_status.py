import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

try:
    from trulens.core.run import INVOCATION_TIMEOUT_IN_MS
    from trulens.core.run import Run
    from trulens.core.run import RunStatus
    from trulens.core.run import SupportedEntryType

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
        # Set default returns for methods used in _compute_overall_computations_status.
        self.base_run_dao.fetch_query_execution_status_by_id.return_value = (
            "SUCCESS"
        )
        self.base_run_dao.fetch_computation_job_results_by_query_id.return_value = pd.DataFrame([
            {
                "METRIC": "answer_relevance",
                "STATUS": "SUCCESS",
                "MESSAGE": "Computed 10 records.",
            }
        ])
        self.base_run_dao.upsert_run_metadata_fields = MagicMock()

        self.fixed_time = 9999999999
        fixed_time = self.fixed_time
        self._orig_get_current_time = Run._get_current_time_in_ms
        Run._get_current_time_in_ms = lambda self: fixed_time

    def tearDown(self):
        Run._get_current_time_in_ms = self._orig_get_current_time

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

    def test_no_completion_status_ingested_records_met(self):
        # Latest invocation has no completion_status.
        # Simulate that the current ingested records count >= input_records_count.
        inv = self.create_invocation("inv1", 1000, 0, None)
        run_metadata = {
            "invocations": {"inv1": inv},
            "metrics": {},
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)
        self.base_run_dao.read_spans_count_from_event_table.return_value = 1000

        status = run._compute_latest_invocation_status(run)
        # Expect upsert to be called and overall status to be INVOCATION_COMPLETED.
        self.assertEqual(status, RunStatus.INVOCATION_COMPLETED)
        self.base_run_dao.upsert_run_metadata_fields.assert_called_once()

    def test_no_completion_status_timeout(self):
        # Latest invocation has no completion_status.
        # Simulate that current ingested count < input_records_count, and elapsed time > INVOCATION_TIMEOUT_IN_MS.
        inv = self.create_invocation("inv1", 1000, 0, None)
        run_metadata = {
            "invocations": {"inv1": inv},
            "metrics": {},
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)
        self.base_run_dao.read_spans_count_from_event_table.return_value = (
            500  # less than 1000
        )

        # If start_time_ms is 1000, then time.time()*1000 must be > (1000 + INVOCATION_TIMEOUT_IN_MS)
        # For example, return_value = (INVOCATION_TIMEOUT_IN_MS / 1000 + 1.1) gives:
        # time.time()*1000 = INVOCATION_TIMEOUT_IN_MS + 1100, which is > (1000 + INVOCATION_TIMEOUT_IN_MS)
        with patch(
            "time.time", return_value=(INVOCATION_TIMEOUT_IN_MS / 1000 + 1.1)
        ):
            status = run._compute_latest_invocation_status(run)
        self.assertEqual(status, RunStatus.INVOCATION_PARTIALLY_COMPLETED)
        self.base_run_dao.upsert_run_metadata_fields.assert_called_once()

    def test_no_completion_status_in_progress(self):
        # Latest invocation has no completion_status.
        # Simulate that current ingested count < input_records_count and elapsed time is less than timeout.
        current_time_ms = self.fixed_time
        inv = self.create_invocation("inv1", current_time_ms - 1000, 0, None)
        run_metadata = {
            "invocations": {"inv1": inv},
            "metrics": {},
            "computations": {},
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)
        self.base_run_dao.read_spans_count_from_event_table.return_value = 500

        # Patch time.time() so that elapsed time is less than INVOCATION_TIMEOUT_IN_MS.
        with patch("time.time", return_value=(current_time_ms / 1000 + 0.5)):
            status = run._compute_latest_invocation_status(run)
        # If no completion_status is set and end_time_ms is 0, we expect INVOCATION_IN_PROGRESS.
        self.assertEqual(status, RunStatus.INVOCATION_IN_PROGRESS)

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

    def test_metrics_not_set_computation_in_progress(self):
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
                    "completion_status": None,  # not set
                    "computation_id": "comp1",
                }
            },
            "computations": {
                "comp1": {
                    "id": "comp1",
                    "query_id": "query1",
                    "start_time_ms": 500,
                    "end_time_ms": None,
                }
            },
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)

        with (
            patch.object(
                self.base_run_dao,
                "fetch_query_execution_status_by_id",
                return_value="IN_PROGRESS",
            ) as mock_fetch_status,
            patch.object(
                self.base_run_dao, "fetch_computation_job_results_by_query_id"
            ) as mock_fetch_results,
            patch.object(
                self.base_run_dao, "upsert_run_metadata_fields"
            ) as mock_upsert,
        ):
            status = run._compute_overall_computations_status(run)
            self.assertEqual(status, RunStatus.COMPUTATION_IN_PROGRESS)
            mock_fetch_status.assert_called_once_with(
                query_start_time_ms=500, query_id="query1"
            )
            mock_upsert.assert_not_called()
            mock_fetch_results.assert_not_called()

    def test_metrics_not_set_computation_success(self):
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
                    "completion_status": None,
                    "computation_id": "comp1",
                }
            },
            "computations": {
                "comp1": {
                    "id": "comp1",
                    "query_id": "query1",
                    "start_time_ms": 500,
                    "end_time_ms": None,
                }
            },
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)

        # Define a side effect for upsert_run_metadata_fields that updates the metric in the run.
        def upsert_side_effect(**kwargs):
            entry_type = kwargs.get("entry_type")
            if entry_type == SupportedEntryType.METRICS.value:
                metric_id = kwargs.get("entry_id")
                new_completion_status = kwargs.get("completion_status")
                # Convert the dict to a Run.CompletionStatus instance if necessary.
                if new_completion_status is not None and isinstance(
                    new_completion_status, dict
                ):
                    new_completion_status = Run.CompletionStatus(
                        **new_completion_status
                    )
                old_metric = run.run_metadata.metrics[metric_id]
                new_metric = old_metric.copy(
                    update={"completion_status": new_completion_status}
                )
                run.run_metadata.metrics[metric_id] = new_metric

        with (
            patch.object(
                self.base_run_dao,
                "fetch_query_execution_status_by_id",
                return_value="SUCCESS",
            ) as mock_fetch_status,
            patch.object(
                self.base_run_dao, "fetch_computation_job_results_by_query_id"
            ) as mock_fetch_results,
            patch.object(
                self.base_run_dao,
                "upsert_run_metadata_fields",
                side_effect=upsert_side_effect,
            ) as mock_upsert,
        ):
            # Simulate computation results returning a row for metric 'answer_relevance'
            df_results = pd.DataFrame([
                {
                    "METRIC": "answer_relevance",
                    "STATUS": "SUCCESS",
                    "MESSAGE": "Computed 10 records.",
                }
            ])
            mock_fetch_results.return_value = df_results

            status = run._compute_overall_computations_status(run)
            self.assertEqual(status, RunStatus.COMPLETED)
            mock_fetch_status.assert_called_once_with(
                query_start_time_ms=500, query_id="query1"
            )
            mock_fetch_results.assert_called_once_with("query1")
            # Expect upsert_run_metadata_fields to be called for both computation and metric update.
            self.assertGreaterEqual(mock_upsert.call_count, 2)

    def test_metrics_not_set_computation_mix_of_success_and_failure(self):
        # Create run metadata with one invocation (COMPLETED) and one metric with no completion status.
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
                    "completion_status": None,  # initially not set
                    "computation_id": "comp1",
                },
                "met2": {
                    "id": "met2",
                    "name": "context_relevance",
                    "completion_status": {
                        "status": "COMPLETED",
                        "record_count": 5,
                    },
                    "computation_id": "comp1",
                },
            },
            "computations": {
                "comp1": {
                    "id": "comp1",
                    "query_id": "query1",
                    "start_time_ms": 500,
                    "end_time_ms": None,
                }
            },
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)

        def upsert_side_effect(**kwargs):
            entry_type = kwargs.get("entry_type")
            if entry_type == SupportedEntryType.METRICS.value:
                metric_id = kwargs.get("entry_id")
                new_completion_status = kwargs.get("completion_status")
                # Convert the dict to a Run.CompletionStatus instance if necessary.
                if new_completion_status is not None and isinstance(
                    new_completion_status, dict
                ):
                    new_completion_status = Run.CompletionStatus(
                        **new_completion_status
                    )
                old_metric = run.run_metadata.metrics[metric_id]
                new_metric = old_metric.copy(
                    update={"completion_status": new_completion_status}
                )
                run.run_metadata.metrics[metric_id] = new_metric

        with (
            patch.object(
                self.base_run_dao,
                "fetch_query_execution_status_by_id",
                return_value="SUCCESS",
            ) as mock_fetch_status,
            patch.object(
                self.base_run_dao,
                "fetch_computation_job_results_by_query_id",
                return_value=pd.DataFrame([
                    {
                        "METRIC": "answer_relevance",
                        "STATUS": "FAILED",
                        "MESSAGE": "Computed 1 records.",
                    }
                ]),
            ) as mock_fetch_results,
            patch.object(
                self.base_run_dao,
                "upsert_run_metadata_fields",
                side_effect=upsert_side_effect,
            ) as mock_upsert,
        ):
            status = run._compute_overall_computations_status(run)
            # Now that the side effect has updated the run's metric completion_status,
            # the final check in _compute_overall_computations_status should detect that
            # all metrics have either FAILED or COMPLETED status and thus return PARTIALLY_COMPLETED overall.
            self.assertEqual(status, RunStatus.PARTIALLY_COMPLETED)
            mock_fetch_status.assert_called_once_with(
                query_start_time_ms=500, query_id="query1"
            )
            mock_fetch_results.assert_called_once_with("query1")
            self.assertGreaterEqual(mock_upsert.call_count, 1)

    def test_metrics_not_set_computation_success_updates_metric_to_failed(self):
        # Create run metadata with one invocation (COMPLETED) and one metric with no completion status.
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
                    "completion_status": None,  # initially not set
                    "computation_id": "comp1",
                },
            },
            "computations": {
                "comp1": {
                    "id": "comp1",
                    "query_id": "query1",
                    "start_time_ms": 500,
                    "end_time_ms": None,
                }
            },
        }
        run = create_dummy_run(run_metadata)
        self.attach_run_dao(run)

        def upsert_side_effect(**kwargs):
            entry_type = kwargs.get("entry_type")
            if entry_type == SupportedEntryType.METRICS.value:
                metric_id = kwargs.get("entry_id")
                new_completion_status = kwargs.get("completion_status")
                # Convert the dict to a Run.CompletionStatus instance if necessary.
                if new_completion_status is not None and isinstance(
                    new_completion_status, dict
                ):
                    new_completion_status = Run.CompletionStatus(
                        **new_completion_status
                    )
                old_metric = run.run_metadata.metrics[metric_id]
                new_metric = old_metric.copy(
                    update={"completion_status": new_completion_status}
                )
                run.run_metadata.metrics[metric_id] = new_metric

        with (
            patch.object(
                self.base_run_dao,
                "fetch_query_execution_status_by_id",
                return_value="SUCCESS",
            ) as mock_fetch_status,
            patch.object(
                self.base_run_dao,
                "fetch_computation_job_results_by_query_id",
                return_value=pd.DataFrame([
                    {
                        "METRIC": "answer_relevance",
                        "STATUS": "FAILED",
                        "MESSAGE": "Computed 1 records.",
                    }
                ]),
            ) as mock_fetch_results,
            patch.object(
                self.base_run_dao,
                "upsert_run_metadata_fields",
                side_effect=upsert_side_effect,
            ) as mock_upsert,
        ):
            status = run._compute_overall_computations_status(run)
            # After the side effect has updated the run's metric completion_status,
            # the final check in _compute_overall_computations_status should detect that
            # all metrics have FAILED status and return FAILED overall.
            self.assertEqual(status, RunStatus.FAILED)
            mock_fetch_status.assert_called_once_with(
                query_start_time_ms=500, query_id="query1"
            )
            mock_fetch_results.assert_called_once_with("query1")
            self.assertGreaterEqual(mock_upsert.call_count, 1)

    def test_metrics_none(self):
        # If run.run_metadata.metrics is None, should return False.
        base_run_metadata = {
            "labels": [],
            "llm_judge_name": None,
            "invocations": {},  # not used in these tests
            "computations": {},
            "metrics": {},
        }
        run = create_dummy_run(base_run_metadata)
        run.run_metadata.metrics = None
        result = run._should_skip_computation("answer_relevance", run)
        self.assertFalse(result)

    def test_no_matching_metric(self):
        # Metrics exist but none match the given metric name.
        base_run_metadata = {
            "labels": [],
            "llm_judge_name": None,
            "invocations": {},
            "computations": {},
            "metrics": {
                "met1": Run.MetricsMetadata.model_validate({
                    "id": "met1",
                    "name": "other_metric",
                    "completion_status": {
                        "status": "COMPLETED",
                        "record_count": 10,
                    },
                    "computation_id": "comp1",
                })
            },
        }
        run = create_dummy_run(base_run_metadata)
        result = run._should_skip_computation("answer_relevance", run)
        self.assertFalse(result)

    def test_one_completed_metric(self):
        # One metric matching "answer_relevance" with COMPLETED status => skip computation.
        base_run_metadata = {
            "labels": [],
            "llm_judge_name": None,
            "invocations": {},
            "computations": {},
            "metrics": {
                "met1": Run.MetricsMetadata.model_validate({
                    "id": "met1",
                    "name": "answer_relevance",
                    "completion_status": {
                        "status": Run.CompletionStatusStatus.COMPLETED,
                        "record_count": 10,
                    },
                    "computation_id": "comp1",
                })
            },
        }
        run = create_dummy_run(base_run_metadata)
        result = run._should_skip_computation("answer_relevance", run)
        self.assertTrue(result)

    def test_one_in_progress_metric(self):
        # One metric matching "answer_relevance" with no completion_status (interpreted as in progress) => skip computation.
        base_run_metadata = {
            "labels": [],
            "llm_judge_name": None,
            "invocations": {},
            "computations": {},
            "metrics": {
                "met1": Run.MetricsMetadata.model_validate({
                    "id": "met1",
                    "name": "answer_relevance",
                    "completion_status": None,
                    "computation_id": "comp1",
                })
            },
        }
        run = create_dummy_run(base_run_metadata)
        result = run._should_skip_computation("answer_relevance", run)
        self.assertTrue(result)

    def test_all_failed_metric(self):
        # One metric matching "answer_relevance" with FAILED status => allow re-computation.
        base_run_metadata = {
            "labels": [],
            "llm_judge_name": None,
            "invocations": {},
            "computations": {},
            "metrics": {
                "met1": Run.MetricsMetadata.model_validate({
                    "id": "met1",
                    "name": "answer_relevance",
                    "completion_status": {
                        "status": Run.CompletionStatusStatus.FAILED,
                        "record_count": 5,
                    },
                    "computation_id": "comp1",
                })
            },
        }
        run = create_dummy_run(base_run_metadata)
        result = run._should_skip_computation("answer_relevance", run)
        self.assertFalse(result)

    def test_multiple_metrics_one_in_progress(self):
        # Multiple metrics with the same name: one FAILED, one in progress => skip (since one is in progress).
        base_run_metadata = {
            "labels": [],
            "llm_judge_name": None,
            "invocations": {},
            "computations": {},
            "metrics": {
                "met1": Run.MetricsMetadata.model_validate({
                    "id": "met1",
                    "name": "answer_relevance",
                    "completion_status": {
                        "status": Run.CompletionStatusStatus.FAILED,
                        "record_count": 5,
                    },
                    "computation_id": "comp1",
                }),
                "met2": Run.MetricsMetadata.model_validate({
                    "id": "met2",
                    "name": "answer_relevance",
                    "completion_status": None,  # in progress
                    "computation_id": "comp2",
                }),
            },
        }
        run = create_dummy_run(base_run_metadata)
        result = run._should_skip_computation("answer_relevance", run)
        self.assertTrue(result)

    def test_multiple_metrics_one_completed(self):
        # Multiple metrics with the same name: one FAILED and one COMPLETED => skip computation.
        base_run_metadata = {
            "labels": [],
            "llm_judge_name": None,
            "invocations": {},
            "computations": {},
            "metrics": {
                "met1": Run.MetricsMetadata.model_validate({
                    "id": "met1",
                    "name": "answer_relevance",
                    "completion_status": {
                        "status": Run.CompletionStatusStatus.FAILED,
                        "record_count": 5,
                    },
                    "computation_id": "comp1",
                }),
                "met2": Run.MetricsMetadata.model_validate({
                    "id": "met2",
                    "name": "answer_relevance",
                    "completion_status": {
                        "status": Run.CompletionStatusStatus.COMPLETED,
                        "record_count": 10,
                    },
                    "computation_id": "comp2",
                }),
            },
        }
        run = create_dummy_run(base_run_metadata)
        result = run._should_skip_computation("answer_relevance", run)
        self.assertTrue(result)
