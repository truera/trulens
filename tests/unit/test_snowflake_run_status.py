from collections import defaultdict
import unittest
from unittest.mock import patch

import pandas as pd
import pytest

# --- Dummy classes to simulate Run metadata (these mimic your proto/Pydantic models) ---


class DummyCompletionStatus:
    def __init__(self, status, record_count=None):
        self.status = status
        self.record_count = record_count

    def model_dump(self):
        return {"status": self.status, "record_count": self.record_count}


class DummyInvocation:
    def __init__(self, id, start_time_ms, completion_status):
        self.id = id
        self.start_time_ms = start_time_ms
        self.completion_status = completion_status


class DummyMetric:
    def __init__(self, id, name, completion_status, computation_id):
        self.id = id
        self.name = name
        self.completion_status = completion_status
        self.computation_id = computation_id


class DummyComputation:
    def __init__(self, id, query_id, start_time_ms):
        self.id = id
        self.query_id = query_id
        self.start_time_ms = start_time_ms


class DummyRunMetadata:
    def __init__(self, invocations=None, metrics=None, computations=None):
        self.invocations = invocations or {}
        self.metrics = metrics or {}
        self.computations = computations or {}


class DummyRun:
    def __init__(
        self, run_metadata, run_name, object_name, object_type, object_version
    ):
        self.run_metadata = run_metadata
        self.run_name = run_name
        self.object_name = object_name
        self.object_type = object_type
        self.object_version = object_version


# --- Dummy Enums for statuses ---
class RunStatus:
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    COMPUTATION_IN_PROGRESS = "COMPUTATION_IN_PROGRESS"


class CompletionStatusStatus:
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    UNKNOWN = "UNKNOWN"


# --- Dummy RunDao with methods to be patched ---
class DummyRunDao:
    def fetch_query_execution_status_by_id(self, query_start_time_ms, query_id):
        return "UNKNOWN"

    def fetch_computation_job_results_by_query_id(self, query_id):
        return pd.DataFrame()

    def upsert_run_metadata_fields(self, **kwargs):
        pass


# --- A dummy Run class for _compute_overall_computations_status ---
class DummyRunClass:
    def __init__(self):
        self.run_dao = DummyRunDao()
        self.run_name = "dummy_run"
        self.object_name = "dummy_agent"
        self.object_type = "EXTERNAL AGENT"
        self.object_version = "v1"

    def _get_current_time_in_ms(self):
        return 9999999999  # fixed dummy value

    def _compute_overall_computations_status(self, run: DummyRun) -> str:
        # Copy of the method to test, using our dummy classes and enums.
        all_existing_metrics = run.run_metadata.metrics.values()

        latest_invocation = max(
            run.run_metadata.invocations.values(),
            key=lambda inv: (inv.start_time_ms or 0, inv.id or ""),
        )
        invocation_completion_status = (
            latest_invocation.completion_status.status
        )

        # Metrics with no set status.
        metrics_status_not_set = [
            metric
            for metric in all_existing_metrics
            if not metric.completion_status
            or not metric.completion_status.status
        ]

        if len(metrics_status_not_set) == 0:
            # All metrics are set.
            if all(
                metric.completion_status.status
                == CompletionStatusStatus.COMPLETED
                for metric in all_existing_metrics
            ):
                return (
                    RunStatus.COMPLETED
                    if invocation_completion_status
                    == CompletionStatusStatus.COMPLETED
                    else RunStatus.PARTIALLY_COMPLETED
                )
            elif all(
                metric.completion_status.status == CompletionStatusStatus.FAILED
                for metric in all_existing_metrics
            ):
                return RunStatus.FAILED
            elif all(
                metric.completion_status.status
                in [
                    CompletionStatusStatus.COMPLETED,
                    CompletionStatusStatus.FAILED,
                ]
                for metric in all_existing_metrics
            ):
                return RunStatus.PARTIALLY_COMPLETED
            return RunStatus.UNKNOWN
        else:
            # Some metrics not set.
            computation_id_to_metrics = defaultdict(list)
            for metric in metrics_status_not_set:
                computation_id_to_metrics[metric.computation_id].append(metric)

            all_computations = run.run_metadata.computations.values()

            some_computation_in_progress = False
            for computation in all_computations:
                if computation.id in computation_id_to_metrics:
                    query_id = computation.query_id
                    query_status = (
                        self.run_dao.fetch_query_execution_status_by_id(
                            query_start_time_ms=computation.start_time_ms,
                            query_id=query_id,
                        )
                    )
                    if query_status == "IN_PROGRESS":
                        some_computation_in_progress = True
                    elif query_status in ("FAILED", "SUCCESS"):
                        self.run_dao.upsert_run_metadata_fields(
                            entry_type="computations",
                            entry_id=computation.id,
                            query_id=query_id,
                            start_time_ms=computation.start_time_ms,
                            end_time_ms=self._get_current_time_in_ms(),
                            run_name=self.run_name,
                            object_name=self.object_name,
                            object_type=self.object_type,
                            object_version=self.object_version,
                        )
                        metrics_in_computation = computation_id_to_metrics[
                            computation.id
                        ]
                        result_rows = self.run_dao.fetch_computation_job_results_by_query_id(
                            query_id
                        )
                        metric_name_to_status = {
                            row["METRIC"]: row["STATUS"]
                            for _, row in result_rows.iterrows()
                        }
                        metric_name_to_computed_records_count = {
                            row["METRIC"]: int(row["MESSAGE"].split(" ")[1])
                            for _, row in result_rows.iterrows()
                        }
                        for metric in metrics_in_computation:
                            if (
                                metric.name in metric_name_to_status
                                and metric.name
                                in metric_name_to_computed_records_count
                            ):
                                self.run_dao.upsert_run_metadata_fields(
                                    entry_type="metrics",
                                    entry_id=metric.id,
                                    computation_id=computation.id,
                                    name=metric.name,
                                    completion_status=DummyCompletionStatus(
                                        status=(
                                            CompletionStatusStatus.COMPLETED
                                            if metric_name_to_status[
                                                metric.name
                                            ]
                                            == "SUCCESS"
                                            else CompletionStatusStatus.FAILED
                                        ),
                                        record_count=metric_name_to_computed_records_count[
                                            metric.name
                                        ],
                                    ).model_dump(),
                                    run_name=self.run_name,
                                    object_name=self.object_name,
                                    object_type=self.object_type,
                                    object_version=self.object_version,
                                )
            if some_computation_in_progress:
                return RunStatus.COMPUTATION_IN_PROGRESS
            else:
                return (
                    RunStatus.COMPLETED
                    if invocation_completion_status
                    == CompletionStatusStatus.COMPLETED
                    else RunStatus.PARTIALLY_COMPLETED
                )


# --- Unit Tests for _compute_overall_computations_status ---
@pytest.mark.snowflake
class TestComputeOverallComputationsStatus(unittest.TestCase):
    def setUp(self):
        self.run_cls = DummyRunClass()

    def test_all_metrics_completed_and_invocation_completed(self):
        # Build a dummy run where:
        # - Latest invocation has COMPLETED status.
        # - All metrics have COMPLETED status.
        invocation = DummyInvocation(
            "inv1",
            1000,
            DummyCompletionStatus(CompletionStatusStatus.COMPLETED),
        )
        metric = DummyMetric(
            "met1",
            "answer_relevance",
            DummyCompletionStatus(
                CompletionStatusStatus.COMPLETED, record_count=10
            ),
            "comp1",
        )
        run_metadata = DummyRunMetadata(
            invocations={"inv1": invocation},
            metrics={"met1": metric},
            computations={},
        )
        run = DummyRun(run_metadata, "run1", "agent", "EXTERNAL AGENT", "v1")
        status = self.run_cls._compute_overall_computations_status(run)
        # Expect overall COMPLETED because invocation and metric statuses are COMPLETED.
        self.assertEqual(status, RunStatus.COMPLETED)

    def test_all_metrics_failed(self):
        # Build a dummy run where all metrics are FAILED.
        invocation = DummyInvocation(
            "inv1",
            1000,
            DummyCompletionStatus(CompletionStatusStatus.COMPLETED),
        )
        metric = DummyMetric(
            "met1",
            "answer_relevance",
            DummyCompletionStatus(
                CompletionStatusStatus.FAILED, record_count=5
            ),
            "comp1",
        )
        run_metadata = DummyRunMetadata(
            invocations={"inv1": invocation},
            metrics={"met1": metric},
            computations={},
        )
        run = DummyRun(run_metadata, "run1", "agent", "EXTERNAL AGENT", "v1")
        status = self.run_cls._compute_overall_computations_status(run)
        self.assertEqual(status, RunStatus.FAILED)

    def test_metrics_mix_completed_failed(self):
        # Build a dummy run where metrics are a mix of COMPLETED and FAILED.
        invocation = DummyInvocation(
            "inv1", 1000, DummyCompletionStatus(CompletionStatusStatus.FAILED)
        )
        metric1 = DummyMetric(
            "met1",
            "answer_relevance",
            DummyCompletionStatus(
                CompletionStatusStatus.COMPLETED, record_count=10
            ),
            "comp1",
        )
        metric2 = DummyMetric(
            "met2",
            "coherence",
            DummyCompletionStatus(
                CompletionStatusStatus.FAILED, record_count=7
            ),
            "comp1",
        )
        run_metadata = DummyRunMetadata(
            invocations={"inv1": invocation},
            metrics={"met1": metric1, "met2": metric2},
            computations={},
        )
        run = DummyRun(run_metadata, "run1", "agent", "EXTERNAL AGENT", "v1")
        status = self.run_cls._compute_overall_computations_status(run)
        # Mixed metrics => overall PARTIALLY_COMPLETED.
        self.assertEqual(status, RunStatus.PARTIALLY_COMPLETED)

    @patch.object(DummyRunDao, "fetch_query_execution_status_by_id")
    @patch.object(DummyRunDao, "fetch_computation_job_results_by_query_id")
    @patch.object(DummyRunDao, "upsert_run_metadata_fields")
    def test_metrics_not_set_computation_in_progress(
        self, mock_upsert, mock_fetch_results, mock_fetch_status
    ):
        # Build a dummy run with one metric that is not set.
        invocation = DummyInvocation(
            "inv1",
            1000,
            DummyCompletionStatus(CompletionStatusStatus.COMPLETED),
        )
        # metric with no completion_status
        metric = DummyMetric("met1", "answer_relevance", None, "comp1")
        computation = DummyComputation("comp1", "query1", 500)
        run_metadata = DummyRunMetadata(
            invocations={"inv1": invocation},
            metrics={"met1": metric},
            computations={"comp1": computation},
        )
        run = DummyRun(run_metadata, "run1", "agent", "EXTERNAL AGENT", "v1")
        # Simulate computation still in progress.
        mock_fetch_status.return_value = "IN_PROGRESS"
        status = self.run_cls._compute_overall_computations_status(run)
        self.assertEqual(status, RunStatus.COMPUTATION_IN_PROGRESS)
        mock_fetch_status.assert_called_once_with(
            query_start_time_ms=500, query_id="query1"
        )
        mock_upsert.assert_not_called()
        mock_fetch_results.assert_not_called()

    @patch.object(DummyRunDao, "fetch_query_execution_status_by_id")
    @patch.object(DummyRunDao, "fetch_computation_job_results_by_query_id")
    @patch.object(DummyRunDao, "upsert_run_metadata_fields")
    def test_metrics_not_set_all_computations_done(
        self, mock_upsert, mock_fetch_results, mock_fetch_status
    ):
        # Build a dummy run with one metric that is not set.
        invocation = DummyInvocation(
            "inv1",
            1000,
            DummyCompletionStatus(CompletionStatusStatus.COMPLETED),
        )
        metric = DummyMetric("met1", "answer_relevance", None, "comp1")
        computation = DummyComputation("comp1", "query1", 500)
        run_metadata = DummyRunMetadata(
            invocations={"inv1": invocation},
            metrics={"met1": metric},
            computations={"comp1": computation},
        )
        run = DummyRun(run_metadata, "run1", "agent", "EXTERNAL AGENT", "v1")
        # Simulate that the computation query finished with SUCCESS.
        mock_fetch_status.return_value = "SUCCESS"
        # Simulate fetch_computation_job_results_by_query_id returning a DataFrame with one row.
        result_data = [
            {
                "METRIC": "answer_relevance",
                "STATUS": "SUCCESS",
                "MESSAGE": "Computed 10 records.",
            }
        ]
        df = pd.DataFrame(result_data)
        mock_fetch_results.return_value = df

        status = self.run_cls._compute_overall_computations_status(run)
        # Since invocation is COMPLETED, overall should be COMPLETED.
        self.assertEqual(status, RunStatus.COMPLETED)
        mock_fetch_status.assert_called_once_with(
            query_start_time_ms=500, query_id="query1"
        )
        mock_fetch_results.assert_called_once_with("query1")
        # Expect upsert_run_metadata_fields to be called for both the computation and the metric update.
        self.assertTrue(mock_upsert.call_count >= 2)


if __name__ == "__main__":
    unittest.main()
