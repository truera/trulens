from __future__ import annotations  # defers evaluation of annotations

from enum import Enum
import inspect
import json
import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Set, Type
import uuid

import pandas as pd
import pydantic
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from trulens.core.utils.json import obj_id_of_obj
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


def _get_all_span_attribute_key_constants(cls: Type, prefix: str) -> List[str]:
    ret = []
    for curr_name in dir(cls):
        if (not curr_name.startswith("__")) and (not curr_name.endswith("__")):
            curr = getattr(cls, curr_name)
            if inspect.isclass(curr):
                ret += _get_all_span_attribute_key_constants(
                    curr, f"{curr_name}"
                )
            elif curr_name == curr_name.upper():
                ret += [f"{prefix + '.' if prefix else ''}{curr_name}"]
    ret = list(set(ret))
    ret = [field for field in ret if not (field.startswith("SpanType"))]

    ret += [
        field.replace(
            "RECORD_ROOT.", ""
        )  # record_root can be optionally specified
        for field in ret
        if field.startswith("RECORD_ROOT.")
    ]
    return ret


def get_all_span_attribute_key_constants() -> set[str]:
    return set(_get_all_span_attribute_key_constants(SpanAttributes, ""))


# Reserved fields (case-insensitive) for dataset specification directly maps to OTEL Span attributes
DATASET_RESERVED_FIELDS: Set[str] = {
    field.lower() for field in get_all_span_attribute_key_constants()
}


INVOCATION_TIMEOUT_IN_MS = (
    5 * 60 * 1000
)  # expected latency from the telemetry pipeline before ingested rows show up in event table


class RunStatus(str, Enum):
    # note this the inferred / computeted status determined by SDK, and is different from the DPO entity level run_status

    # invocation statuses
    INVOCATION_IN_PROGRESS = "INVOCATION_IN_PROGRESS"
    INVOCATION_COMPLETED = "INVOCATION_COMPLETED"
    INVOCATION_PARTIALLY_COMPLETED = "INVOCATION_PARTIALLY_COMPLETED"

    # computation statuses
    COMPUTATION_IN_PROGRESS = "COMPUTATION_IN_PROGRESS"
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"

    CREATED = "CREATED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"  # TODO: do we want to show this to users?


class SupportedEntryType(str, Enum):
    INVOCATIONS = "invocations"
    COMPUTATIONS = "computations"
    METRICS = "metrics"


SUPPORTED_ENTRY_TYPES = [
    SupportedEntryType.INVOCATIONS,
    SupportedEntryType.COMPUTATIONS,
    SupportedEntryType.METRICS,
]


def validate_dataset_spec(
    dataset_spec: Dict[str, str],
) -> Dict[str, str]:
    """
    Validates and normalizes the dataset column specification to ensure it contains only
    currently supported span attributes and that the keys are in the correct format.

    Args:
        dataset_spec: The user-provided dictionary with column names.

    Returns:
        A validated and normalized dictionary.

    Raises:
        ValueError: If any invalid field is present.
    """

    normalized_spec = {}

    for key, value in dataset_spec.items():
        normalized_key = key.lower()

        # Ensure that the key is one of the valid reserved fields or its subscripted form
        if not any(
            normalized_key.startswith(reserved_field)
            for reserved_field in DATASET_RESERVED_FIELDS
        ):
            raise ValueError(f"Invalid field '{key}' found in dataset_spec.")

        # Add the normalized field to the dictionary
        normalized_spec[normalized_key] = value

    return normalized_spec


class RunConfig(BaseModel):
    run_name: str = Field(
        ...,
        description="Unique name of the run. This name should be unique within the object.",
    )
    dataset_name: str = Field(
        default=...,
        description="Mandatory field. The name of a user's Snowflake Table / View  (e.g. 'user_table_name_1'), or any user specified name of input dataframe.",
    )

    source_type: str = Field(
        default="DATAFRAME",
        description="Type of the source (e.g. 'DATAFRAME' for user provided dataframe or 'TABLE' for user table in Snowflake).",
    )

    dataset_spec: Dict[str, str] = Field(
        default=...,
        description="Mandatory column name mapping from reserved dataset fields to column names in user's table.",
    )

    description: Optional[str] = Field(
        default=None, description="A description for the run."
    )
    label: Optional[str] = Field(
        default=None,
        description="Text label to group the runs. Take a single label for now",
    )
    llm_judge_name: Optional[str] = Field(
        default=None,
        description="Name of the LLM judge to be used for the run.",
    )


class Run(BaseModel):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",  # allow custom obj like RunDao to be passed as a parameter and more importantly, account for
        # additional fields in Run metadata JSON response.
    )

    """
    Run class for managing run state / attributes in the SDK client.

    This model is meant to be used and accessed through
    methods like describe() (which uses the underlying RunDao) to obtain the run metadata.
    """

    run_dao: Any = Field(
        ..., description="DAO instance for run operations.", exclude=True
    )

    app: Any = Field(
        ...,
        description="TruLens app/recorder instance to be invoked during run.",
        exclude=True,
    )

    main_method_name: str = Field(
        ..., description="Main method of the app.", exclude=True
    )

    tru_session: Any = Field(
        ..., description="TruSession instance.", exclude=True
    )
    object_name: str = Field(
        ...,
        description="Name of the managing object (e.g. name of 'EXTERNAL AGENT').",
    )

    object_type: str = Field(
        ..., description="Type of the managing object (e.g. 'EXTERNAL AGENT')."
    )

    object_version: Optional[str] = Field(
        default=None, description="Version of the managing object."
    )

    run_name: str = Field(
        ...,
        description="Unique name of the run. This name should be unique within the object.",
    )

    description: Optional[str] = Field(
        default=None, description="A description for the run."
    )

    class RunMetadata(BaseModel):
        labels: List[Optional[str]] = Field(
            default=[],
            description="Text label to group the runs. Take a single label for now",
        )
        llm_judge_name: Optional[str] = Field(
            default=None,
            description="Name of the LLM judge to be used for the run.",
        )
        invocations: Optional[Dict[str, Run.InvocationMetadata]] = Field(
            default=None,
            description="Map of invocation metadata with invocation ID as key.",
        )
        computations: Optional[Dict[str, Run.ComputationMetadata]] = Field(
            default=None,
            description="Map of computation metadata with computation ID as key.",
        )
        metrics: Optional[Dict[str, Run.MetricsMetadata]] = Field(
            default=None,
            description="Map of metrics metadata with metric ID as key.",
        )

    run_metadata: RunMetadata = Field(
        default=...,
        description="Run metadata that maintains states needed for app invocation and metrics computation.",
    )

    class SourceInfo(BaseModel):
        name: str = Field(
            ...,
            description="Name of the source (e.g. name of the table).",
        )
        column_spec: Dict[str, str] = Field(
            default=...,
            description="Column name mapping from reserved dataset fields to column names in user's table.",
        )
        source_type: str = Field(
            default="DATAFRAME",
            description="Type of the source (e.g. 'DATAFRAME').",
        )

    source_info: SourceInfo = Field(
        default=...,
        description="Source information for the run.",
    )

    class CompletionStatusStatus(str, Enum):
        UNKNOWN = "UNKNOWN"
        PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
        COMPLETED = "COMPLETED"
        FAILED = "FAILED"

    class CompletionStatus(BaseModel):
        status: Run.CompletionStatusStatus = Field(
            ..., description="The status of the completion."
        )
        record_count: Optional[int] = Field(
            default=None, description="The count of records processed."
        )
        model_config = ConfigDict(json_encoders={Enum: lambda o: o.value})

    class InvocationMetadata(BaseModel):
        input_records_count: Optional[int] = Field(
            default=None,
            description="The number of input records in the dataset.",
        )
        id: Optional[str] = Field(
            default=None,
            description="The unique identifier for the invocation metadata.",
        )
        start_time_ms: Optional[int] = Field(
            default=None,
            description="The start time of the invocation.",
        )
        end_time_ms: Optional[int] = Field(
            default=None,
            description="The end time of the invocation.",
        )
        completion_status: Optional[Run.CompletionStatus] = Field(
            default=None,
            description="The status of the invocation.",
        )

    class ComputationMetadata(BaseModel):
        id: Optional[str] = Field(
            default=None,
            description="Unique id, even if name is repeated.",
        )
        query_id: Optional[str] = Field(
            default=None,
            description="Query id associated with metric computation.",
        )
        start_time_ms: Optional[int] = Field(
            default=None,
            description="Start time of the computation in milliseconds.",
        )
        end_time_ms: Optional[int] = Field(
            default=None,
            description="End time of the computation in milliseconds.",
        )

    class MetricsMetadata(BaseModel):
        id: Optional[str] = Field(
            default=None,
            description="Unique id for the metrics metadata.",
        )
        name: Optional[str] = Field(
            default=None,
            description="Name of the metric.",
        )
        completion_status: Optional[Run.CompletionStatus] = Field(
            default=None,
            description="Completion status of the metric computation.",
        )
        computation_id: Optional[str] = Field(
            default=None,
            description="ID of the computation associated with the metric.",
        )

    def describe(self) -> dict:
        """
        Retrieve the metadata of the Run object.
        """

        run_metadata_df = self.run_dao.get_run(
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
        )
        if run_metadata_df.empty:
            raise ValueError(f"Run {self.run_name} not found.")

        return json.loads(
            list(run_metadata_df.to_dict(orient="records")[0].values())[0]
        )

    def delete(self) -> None:
        """
        Delete the run by its name and object name.
        """
        self.run_dao.delete_run(
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
        )

    def _can_start_new_invocation(self, current_run_status: RunStatus) -> bool:
        """
        Check if the run is in a state that allows starting a new invocation.
        """
        return current_run_status in [
            RunStatus.CREATED,
            RunStatus.INVOCATION_PARTIALLY_COMPLETED,
            RunStatus.FAILED,
            RunStatus.UNKNOWN,
        ]

    def _can_start_new_metric_computation(
        self, current_run_status: RunStatus
    ) -> bool:
        """
        Check if the run is in a state that allows starting a new metric computation.
        """
        if current_run_status == RunStatus.COMPUTATION_IN_PROGRESS:
            logger.warning(
                "Previous computation(s) still in progress. Starting another new metric computation when computation is in progress."
            )
        # TODO: verify if we allow starting a new compute query when in-progress computation exists
        return current_run_status in [
            RunStatus.INVOCATION_COMPLETED,
            RunStatus.INVOCATION_PARTIALLY_COMPLETED,
            RunStatus.COMPUTATION_IN_PROGRESS,
            RunStatus.COMPLETED,
            RunStatus.PARTIALLY_COMPLETED,
            RunStatus.FAILED,
        ]

    def _is_invocation_started(self, run: Run) -> bool:
        return (
            run.run_metadata.invocations is not None
            and len(run.run_metadata.invocations) > 0
        )

    def _compute_latest_invocation_status(self, run: Run) -> RunStatus:
        latest_invocation = max(
            run.run_metadata.invocations.values(),
            key=lambda inv: (inv.start_time_ms or 0, inv.id or ""),
        )

        if (
            latest_invocation.completion_status
            and latest_invocation.completion_status.status
        ):
            completion_status = latest_invocation.completion_status.status
            if completion_status == Run.CompletionStatusStatus.COMPLETED:
                return RunStatus.INVOCATION_COMPLETED
            elif (
                completion_status
                == Run.CompletionStatusStatus.PARTIALLY_COMPLETED
            ):
                return RunStatus.INVOCATION_PARTIALLY_COMPLETED
            elif completion_status == Run.CompletionStatusStatus.FAILED:
                return RunStatus.FAILED
            else:
                logger.warning(
                    f"Unknown completion status {completion_status} for invocation {latest_invocation.id}"
                )
                return RunStatus.UNKNOWN

        current_ingested_records_count = (
            self.run_dao.read_spans_count_from_event_table(
                object_name=self.object_name,
                run_name=self.run_name,
                span_type="record_root",
            )
        )
        logger.info(
            f"Current ingested records count: {current_ingested_records_count}"
        )

        if (
            latest_invocation.input_records_count
            and current_ingested_records_count
            == latest_invocation.input_records_count
        ):
            # happy case, add end time and update status
            self.run_dao.upsert_run_metadata_fields(
                entry_type=SupportedEntryType.INVOCATIONS,
                entry_id=latest_invocation.id,
                input_records_count=latest_invocation.input_records_count,
                start_time_ms=latest_invocation.start_time_ms,
                end_time_ms=self._get_current_time_in_ms(),
                completion_status=Run.CompletionStatus(
                    status=Run.CompletionStatusStatus.COMPLETED,
                    record_count=current_ingested_records_count,
                ).model_dump(),
                run_name=self.run_name,
                object_name=self.object_name,
                object_type=self.object_type,
                object_version=self.object_version,
            )

            return RunStatus.INVOCATION_COMPLETED

        elif (
            latest_invocation.start_time_ms
            and time.time() * 1000 - latest_invocation.start_time_ms
            > INVOCATION_TIMEOUT_IN_MS
        ):
            # inconclusive case, timeout reached and add end time and update completion status in DPO
            logger.warning("Invocation timeout reached and concluded")
            self.run_dao.upsert_run_metadata_fields(
                entry_type=SupportedEntryType.INVOCATIONS,
                entry_id=latest_invocation.id,
                input_records_count=latest_invocation.input_records_count,
                start_time_ms=latest_invocation.start_time_ms,
                end_time_ms=self._get_current_time_in_ms(),
                completion_status=Run.CompletionStatus(
                    status=Run.CompletionStatusStatus.PARTIALLY_COMPLETED,
                    record_count=current_ingested_records_count,
                ).model_dump(),
                run_name=self.run_name,
                object_name=self.object_name,
                object_type=self.object_type,
                object_version=self.object_version,
            )

            return RunStatus.INVOCATION_PARTIALLY_COMPLETED

        else:
            return (
                RunStatus.INVOCATION_IN_PROGRESS
                if latest_invocation.end_time_ms == 0
                else RunStatus.UNKNOWN
            )

    def _is_computation_started(self, run: Run) -> bool:
        return (
            run.run_metadata.computations is not None
            and len(run.run_metadata.computations) > 0
        )

    def _compute_overall_computations_status(self, run: Run) -> RunStatus:
        all_computations = run.run_metadata.computations.values()

        latest_invocation = max(
            run.run_metadata.invocations.values(),
            key=lambda inv: (inv.start_time_ms or 0, inv.id or ""),
        )
        invocation_completion_status = (
            latest_invocation.completion_status.status
        )

        computation_sproc_query_ids_to_query_status = {}
        for computation in all_computations:
            query_id = computation.query_id
            query_status = self.run_dao.fetch_query_execution_status_by_id(
                query_start_time_ms=computation.start_time_ms,
                query_id=query_id,
            )

            if query_status == "IN_PROGRESS":
                logger.info(
                    f"Computation {computation.id} is still running or being queued."
                )
                # Returning early to avoid unnecessary checks as long as at least one computation is still running
                return RunStatus.COMPUTATION_IN_PROGRESS
            computation_sproc_query_ids_to_query_status[query_id] = query_status
        # all computations are done, update DPO computations metadata
        for computation in all_computations:
            self.run_dao.upsert_run_metadata_fields(
                entry_type=SupportedEntryType.COMPUTATIONS,
                entry_id=computation.id,
                query_id=computation.query_id,
                start_time_ms=computation.start_time_ms,
                end_time_ms=self._get_current_time_in_ms(),
                run_name=self.run_name,
                object_name=self.object_name,
                object_type=self.object_type,
                object_version=self.object_version,
            )

        if all(
            status == "SUCCESS"
            for status in computation_sproc_query_ids_to_query_status.values()
        ):
            overall_computation_status = (
                RunStatus.COMPLETED
                if invocation_completion_status
                == Run.CompletionStatusStatus.COMPLETED
                else RunStatus.PARTIALLY_COMPLETED
            )

            for computation in all_computations:
                query_id = computation.query_id

                result_rows = (
                    self.run_dao.fetch_computation_job_results_by_query_id(
                        query_id
                    )
                )
                for i, row in result_rows.iterrows():
                    row_msg = row["MESSAGE"]

                    computed_records_count = int(
                        row_msg.split(" ")[1]
                    )  # TODO change to regex or directly read the field when available
                    metric_status = row["STATUS"]
                    metric_metatada_id = str(uuid.uuid4())
                    if metric_status == "SUCCESS":
                        logger.info(
                            f"Metrics computation for {row['METRIC']} succeeded."
                        )

                        self.run_dao.upsert_run_metadata_fields(
                            entry_type=SupportedEntryType.METRICS,
                            entry_id=metric_metatada_id,
                            computation_id=computation.id,
                            name=row["METRIC"],
                            completion_status=Run.CompletionStatus(
                                status=Run.CompletionStatusStatus.COMPLETED,
                                record_count=computed_records_count,
                            ).model_dump(),
                            run_name=self.run_name,
                            object_name=self.object_name,
                            object_type=self.object_type,
                            object_version=self.object_version,
                        )

                    elif metric_status == "FAILURE":
                        logger.warning(
                            f"Metrics computation for {row['METRIC']} failed."
                        )

                        self.run_dao.upsert_run_metadata_fields(
                            entry_type=SupportedEntryType.METRICS,
                            entry_id=metric_metatada_id,
                            computation_id=computation.id,
                            name=row["METRIC"],
                            completion_status=Run.CompletionStatus(
                                status=Run.CompletionStatusStatus.FAILED,
                                record_count=computed_records_count,
                            ).model_dump(),
                            run_name=self.run_name,
                            object_name=self.object_name,
                            object_type=self.object_type,
                            object_version=self.object_version,
                        )

            return overall_computation_status
        if all(
            status == "FAILED"
            for status in computation_sproc_query_ids_to_query_status.values()
        ):
            logger.warning("All computations failed.")
            return RunStatus.FAILED
        if (
            all(
                status == "SUCCESS" or status == "FAILED"
                for status in computation_sproc_query_ids_to_query_status.values()
            )
            and self.run_dao.read_spans_count_from_event_table(
                object_name=self.object_name,
                run_name=self.run_name,
                span_type="eval_root",
            )
            == 0
        ):
            # metric (eval) result ingestion failed for some reason
            logger.warning(
                "All computations concluded, but no metric results ingested."
            )
            return RunStatus.FAILED
        if (
            invocation_completion_status == Run.CompletionStatusStatus.COMPLETED
            or invocation_completion_status
            == Run.CompletionStatusStatus.PARTIALLY_COMPLETED
        ) and not any(
            status == "IN_PROGRESS"
            for status in computation_sproc_query_ids_to_query_status.values()
        ):
            return RunStatus.PARTIALLY_COMPLETED

        logger.warning("Cannot determine run status")

        return RunStatus.UNKNOWN

    def start(self, input_df: Optional[pd.DataFrame] = None):
        """
        Start the run by invoking the main method of the user's app with the input data

        Args:
            input_df (Optional[pd.DataFrame], optional): user provided input dataframe.
        """
        current_status = self.get_status()
        logger.info(f"Current run status: {current_status}")
        if not self._can_start_new_invocation(current_status):
            return f"Cannot start a new invocation when in run status: {current_status}. Valid statuses are: {RunStatus.CREATED}, {RunStatus.INVOCATION_PARTIALLY_COMPLETED}, or {RunStatus.FAILED}."

        if input_df is None:
            logger.info(
                "No input dataframe provided. Fetching input data from source."
            )
            # TODO: update the source_info.source_type to 'TABLE'
            rows = self.run_dao.session.sql(
                f"SELECT * FROM {self.source_info.name}"
            ).collect()
            input_df = pd.DataFrame([row.as_dict() for row in rows])

        dataset_spec = self.source_info.column_spec

        # Preprocess the dataset_spec to create mappings for input columns
        # and map the inputs for reserved fields only once, before the iteration over rows.

        reserved_field_column_mapping = {}

        # Process dataset column spec to handle subscripting logic for input columns
        for reserved_field, user_column in dataset_spec.items():
            reserved_field_column_mapping[reserved_field] = user_column

        input_records_count = len(input_df)

        invocation_metadata_id = self.run_dao._compute_invocation_metadata_id(
            dataset_name=self.source_info.name,
            input_records_count=input_records_count,
        )
        start_time_ms = self._get_current_time_in_ms()

        logger.info(
            f"Creating or updating invocation metadata with {input_records_count} records from input."
        )

        self.run_dao.upsert_run_metadata_fields(
            entry_type=SupportedEntryType.INVOCATIONS,
            entry_id=invocation_metadata_id,
            start_time_ms=start_time_ms,
            end_time_ms=0,  # required field
            run_name=self.run_name,
            input_records_count=input_records_count,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
        )

        # user app invocation - will block until the app completes
        try:
            for i, row in input_df.iterrows():
                main_method_args = []

                # Call the instrumented main method with the arguments
                # TODO (dhuang) better way to check span attributes, also is this all we need to support?
                input_id = (
                    row[dataset_spec["input_id"]]
                    if "input_id" in dataset_spec
                    else None
                )
                input_col = None
                if input_id is None:
                    if "input" in dataset_spec:
                        input_col = dataset_spec["input"]
                    elif "record_root.input" in dataset_spec:
                        input_col = dataset_spec["record_root.input"]
                    if input_col:
                        input_id = obj_id_of_obj(row[input_col])
                        main_method_args.append(row[input_col])

                ground_truth_output = row.get(
                    dataset_spec.get("ground_truth_output")
                    or dataset_spec.get("record_root.ground_truth_output")
                )

                self.app.instrumented_invoke_main_method(
                    run_name=self.run_name,
                    input_id=input_id,
                    ground_truth_output=ground_truth_output,
                    main_method_args=tuple(
                        main_method_args
                    ),  # Ensure correct order
                    main_method_kwargs=None,  # don't take any kwargs for now so we don't break TruChain / TruLlama where input argument name cannot be defined by users.
                )
        except Exception as e:
            logger.exception(
                f"Error encountered during invoking app main method: {e}."
            )

            self.run_dao.upsert_run_metadata_fields(
                entry_type=SupportedEntryType.INVOCATIONS,
                entry_id=invocation_metadata_id,
                start_time_ms=start_time_ms,
                input_records_count=input_records_count,
                end_time_ms=self._get_current_time_in_ms(),
                completion_status=Run.CompletionStatus(
                    status=Run.CompletionStatusStatus.FAILED,
                ).model_dump(),
                run_name=self.run_name,
                object_name=self.object_name,
                object_type=self.object_type,
                object_version=self.object_version,
            )

            raise

        self.tru_session.force_flush()
        logger.info("Run started, invocation done and ingestion in process.")

    def _get_current_time_in_ms(self) -> int:
        return int(round(time.time() * 1000))

    def get_status(self) -> RunStatus:
        # first read from DPO backend, and decide if we need to update status.
        # TODO: potentially an expensive call, and should optimize in the near future.
        run_metadata_df = self.run_dao.get_run(
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
        )

        run = Run.from_metadata_df(
            run_metadata_df,
            {
                "app": self,
                "main_method_name": self.main_method_name,
                "run_dao": self.run_dao,
                "tru_session": self.tru_session,
            },
        )

        if not self._is_invocation_started(run):
            logger.info("Run is created, no invocation nor computation yet.")
            return RunStatus.CREATED
        elif self._is_computation_started(run):
            logger.info(
                "Run is created, invocation done, and computation started."
            )
            return self._compute_overall_computations_status(run)

        else:
            logger.info("Run is created, invocation started.")
            return self._compute_latest_invocation_status(run)

    def compute_metrics(self, metrics: List[str]) -> str:
        run_status = self.get_status()

        logger.info(f"Current run status: {run_status}")
        if not self._can_start_new_metric_computation(run_status):
            return f"""Cannot start a new metric computation when in run status: {run_status}. Valid statuses are: {RunStatus.INVOCATION_COMPLETED}, {RunStatus.INVOCATION_PARTIALLY_COMPLETED},
        {RunStatus.COMPUTATION_IN_PROGRESS}, {RunStatus.COMPLETED}, {RunStatus.PARTIALLY_COMPLETED}, {RunStatus.FAILED}."""

        async_job = self.run_dao.call_compute_metrics_query(
            metrics=metrics,
            object_name=self.object_name,
            object_version=self.object_version,
            object_type=self.object_type,
            run_name=self.run_name,
        )

        query_id = async_job.query_id

        logger.info(f"Query id for metrics computation: {query_id}")

        computation_metadata_id = str(uuid.uuid4())

        computation_start_time_ms = self._get_current_time_in_ms()

        self.run_dao.upsert_run_metadata_fields(
            entry_type=SupportedEntryType.COMPUTATIONS,
            entry_id=computation_metadata_id,
            query_id=query_id,
            start_time_ms=computation_start_time_ms,
            end_time_ms=0,
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
        )

        logger.info("Metrics computation job started")
        return "Metrics computation in progress."

    def cancel(self):
        raise NotImplementedError("cancel is not implemented yet.")

    def update(
        self, description: Optional[str] = None, label: Optional[str] = None
    ):
        """
        Only description and label are allowed to be updated at the moment.
        """
        raise NotImplementedError("update is not implemented yet.")

    @classmethod
    def from_metadata_df(
        cls, metadata_df: pd.DataFrame, extra: Dict[str, Any]
    ) -> Run:
        """
        Create a Run instance from a metadata DataFrame returned by the DAO,
        and enrich it with additional fields (which are not persisted on the server).

        Args:
            metadata_df: A pandas DataFrame containing run metadata.
                We assume the first row contains a JSON string in its first cell.
            extra: A dictionary of extra fields to add, such as:
                {
                    "app": <app instance>,
                    "main_method_name": <method name>,
                    "run_dao": <dao instance>,
                    "object_name": <object name>,
                    "object_type": <object type>
                }

        Returns:
            A validated Run instance.
        """
        if metadata_df.empty:
            raise ValueError("No run metadata found.")

        # Assume the first cell of the first row contains the JSON string.

        metadata_str = metadata_df.iloc[0].values[0]
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            raise ValueError(
                "The first cell of the first row does not contain a valid JSON string."
            )

        metadata.update(extra)

        return cls.model_validate(metadata)
