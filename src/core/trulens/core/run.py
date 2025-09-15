from __future__ import annotations  # defers evaluation of annotations

from enum import Enum
import inspect
import json
import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Set, Type

import pandas as pd
import pydantic
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_serializer
from trulens.core.utils.json import obj_id_of_obj
from trulens.otel.semconv.trace import ResourceAttributes
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


EXPECTED_TELEMETRY_LATENCY_IN_MS = (
    2 * 60 * 1000
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
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class SupportedEntryType(str, Enum):
    INVOCATIONS = "invocations"
    COMPUTATIONS = "computations"
    METRICS = "metrics"


SUPPORTED_ENTRY_TYPES = [e.value for e in SupportedEntryType]


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

    run_status: Optional[str] = Field(
        default=None,
        description="The status of the run on the entity level. Currently it can only be ACTIVE or CANCELLED.",
    )

    description: Optional[str] = Field(
        default=None, description="A description for the run."
    )

    class RunMetadata(BaseModel):
        labels: Optional[List[str]] = Field(
            default=None,
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
        STARTED = "STARTED"
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

        @field_serializer("status")
        def serialize_status(self, status: Run.CompletionStatusStatus, _info):
            return status.value

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

        raw_json = json.loads(
            list(run_metadata_df.to_dict(orient="records")[0].values())[0]
        )

        # remove / hide entity-level run_status field to avoid customer's confusion
        raw_json.pop("run_status", None)
        return raw_json

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
        if self._is_cancelled():
            logger.warning("Cannot start a new invocation for a cancelled run.")
            return False

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
        if self._is_cancelled():
            logger.warning(
                "Cannot start a new metric computation for a cancelled run."
            )
            return False

        if current_run_status == RunStatus.COMPUTATION_IN_PROGRESS:
            logger.warning(
                "Previous computation(s) still in progress. Starting another new metric computation when computation is in progress."
            )

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
            elif completion_status == Run.CompletionStatusStatus.STARTED:
                return RunStatus.INVOCATION_IN_PROGRESS
            elif completion_status == Run.CompletionStatusStatus.FAILED:
                return RunStatus.FAILED
            else:
                logger.warning(
                    f"Unknown completion status {completion_status} for invocation {latest_invocation.id}"
                )
                return RunStatus.UNKNOWN

    def _metrics_computation_started(self, run: Run) -> bool:
        return (
            run.run_metadata.metrics is not None
            and len(run.run_metadata.metrics) > 0
        )

    def _resolve_overall_metrics_status(
        self, all_metrics, invocation_completion_status
    ):
        """
        Given the list of all metrics, resolve and return the overall metrics status.
        """
        if all(
            metric.completion_status
            and metric.completion_status.status
            == Run.CompletionStatusStatus.COMPLETED
            for metric in all_metrics
        ):
            return (
                RunStatus.COMPLETED
                if invocation_completion_status
                == Run.CompletionStatusStatus.COMPLETED
                else RunStatus.PARTIALLY_COMPLETED
            )
        elif all(
            metric.completion_status
            and metric.completion_status.status
            == Run.CompletionStatusStatus.FAILED
            for metric in all_metrics
        ):
            return RunStatus.FAILED
        elif all(
            metric.completion_status
            and metric.completion_status.status
            in [
                Run.CompletionStatusStatus.COMPLETED,
                Run.CompletionStatusStatus.FAILED,
            ]
            for metric in all_metrics
        ):
            return RunStatus.PARTIALLY_COMPLETED
        elif any(
            metric.completion_status.status
            == Run.CompletionStatusStatus.STARTED
            for metric in all_metrics
        ):
            return RunStatus.COMPUTATION_IN_PROGRESS
        else:
            logger.warning(
                "Cannot determine run status. Metrics: %s",
                [metric.name for metric in all_metrics],
            )
            return RunStatus.UNKNOWN

    def _compute_overall_computations_status(self, run: Run) -> RunStatus:
        all_existing_metrics = run.run_metadata.metrics.values()

        latest_invocation = max(
            run.run_metadata.invocations.values(),
            key=lambda inv: (inv.start_time_ms or 0, inv.id or ""),
        )
        invocation_completion_status = (
            latest_invocation.completion_status.status
        )

        # check all metrics and see if their completion status are set or not
        metrics_status_not_set = [
            metric
            for metric in all_existing_metrics
            if not metric.completion_status
            or not metric.completion_status.status
        ]

        if len(metrics_status_not_set) != 0:
            logger.warning(
                f"Metrics status not set for: {[metric.name for metric in metrics_status_not_set]}."
            )
            raise ValueError(
                f"Metrics status not set for: {[metric.name for metric in metrics_status_not_set]}."
            )

        logger.info("All metrics statuses are set as expected.")
        return self._resolve_overall_metrics_status(
            all_existing_metrics, invocation_completion_status
        )

    def start(
        self, input_df: Optional[pd.DataFrame] = None, virtual: bool = False
    ):
        """
        Start the run by invoking the main method of the user's app with the input data

        Args:
            input_df (Optional[pd.DataFrame], optional): user provided input dataframe.
            virtual (bool, optional): If True, creates OTEL spans from existing data without app invocation.
        """
        current_status = self.get_status()
        logger.info(f"Current run status: {current_status}")
        if not self._can_start_new_invocation(current_status):
            return f"Cannot start a new invocation when in run status: {current_status}. Valid statuses are: {RunStatus.CREATED}, {RunStatus.INVOCATION_PARTIALLY_COMPLETED}, or {RunStatus.FAILED}."

        if input_df is None:
            logger.info(
                "No input dataframe provided. Fetching input data from source."
            )
            rows = self.run_dao.session.sql(
                f"SELECT * FROM {self.source_info.name}"
            ).collect()
            input_df = pd.DataFrame([row.as_dict() for row in rows])

        dataset_spec = self.source_info.column_spec

        input_records_count = len(input_df)

        logger.info(
            f"Creating or updating invocation metadata with {input_records_count} records from input."
        )

        # user app invocation - will block until the app completes
        try:
            if virtual:
                self._create_virtual_spans(
                    input_df, dataset_spec, input_records_count
                )
            else:
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

                    # Extract additional method arguments from dataset_spec
                    # Skip fields that are already handled or are special metadata fields
                    special_fields = {
                        "input_id",
                        "input",
                        "record_root.input",
                        "ground_truth_output",
                        "record_root.ground_truth_output",
                    }

                    for spec_key, column_name in dataset_spec.items():
                        if (
                            spec_key not in special_fields
                            and column_name in row
                        ):
                            main_method_args.append(row[column_name])

                    ground_truth_output = row.get(
                        dataset_spec.get("ground_truth_output")
                        or dataset_spec.get("record_root.ground_truth_output")
                    )

                    self.app.instrumented_invoke_main_method(
                        run_name=self.run_name,
                        input_id=input_id,
                        input_records_count=input_records_count,
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
            raise

        self.tru_session.force_flush()
        logger.info(
            f"Flushed all spans for the run {self.run_name}; and exported them to the telemetry pipeline."
        )

        # we start the ingestion sproc after the app invocation is done, so that
        # app invocation time does not count toward the ingestion timeout set on the task orchestration layer.
        self.run_dao.start_ingestion_query(
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
            run_name=self.run_name,
            input_records_count=input_records_count,
        )
        logger.info("Run started, invocation done and ingestion in process.")

    def _create_virtual_spans(
        self,
        input_df: pd.DataFrame,
        dataset_spec: Dict[str, str],
        input_records_count: int,
    ):
        """
        Create OTEL spans from existing data without actual app invocation.
        This method replicates the span structure that would be created by instrumented app methods,
        but uses existing data from the input DataFrame.
        """
        from trulens.core.otel.instrument import OtelRecordingContext

        logger.info(f"Creating virtual spans for {len(input_df)} records")

        for i, row in input_df.iterrows():
            # Extract input information similar to the regular flow
            input_id = (
                row[dataset_spec["input_id"]]
                if "input_id" in dataset_spec
                else None
            )

            # Determine input column and value
            input_col = None
            input_value = None
            if input_id is None:
                if "input" in dataset_spec:
                    input_col = dataset_spec["input"]
                elif "record_root.input" in dataset_spec:
                    input_col = dataset_spec["record_root.input"]
                if input_col and input_col in row:
                    input_value = row[input_col]
                    input_id = obj_id_of_obj(input_value)

            # Extract ground truth output
            ground_truth_output = row.get(
                dataset_spec.get("ground_truth_output")
                or dataset_spec.get("record_root.ground_truth_output")
            )

            # Extract output value (assuming it exists in the data)
            output_value = None
            if "output" in dataset_spec and dataset_spec["output"] in row:
                output_value = row[dataset_spec["output"]]
            elif (
                "record_root.output" in dataset_spec
                and dataset_spec["record_root.output"] in row
            ):
                output_value = row[dataset_spec["record_root.output"]]

            # Extract contexts if available
            contexts = None
            if (
                "retrieved_contexts" in dataset_spec
                and dataset_spec["retrieved_contexts"] in row
            ):
                contexts_str = row[dataset_spec["retrieved_contexts"]]
                if contexts_str:
                    contexts = [ctx.strip() for ctx in contexts_str.split(",")]

            # Create OTEL recording context similar to instrumented_invoke_main_method
            # For virtual runs, check if the app has a real underlying app or is a placeholder
            if hasattr(self.app, "app") and self.app.app is None:
                # This is a virtual TruApp with no underlying app - create spans manually
                self._create_virtual_spans_without_app(
                    input_id,
                    input_value,
                    output_value,
                    contexts,
                    input_records_count,
                    ground_truth_output,
                )
            else:
                # Regular TruApp with real app - use normal recording context
                with OtelRecordingContext(
                    tru_app=self.app,
                    app_name=self.app.app_name,
                    app_version=self.app.app_version,
                    run_name=self.run_name,
                    input_id=input_id,
                    input_records_count=input_records_count,
                    ground_truth_output=ground_truth_output,
                ):
                    self._create_virtual_spans_in_context(
                        input_value, output_value, contexts
                    )

    def _create_virtual_spans_in_context(
        self, input_value, output_value, contexts
    ):
        """Create virtual spans within an existing OTEL recording context."""
        from opentelemetry import trace
        from trulens.experimental.otel_tracing.core.session import (
            TRULENS_SERVICE_NAME,
        )

        tracer = trace.get_tracer_provider().get_tracer(TRULENS_SERVICE_NAME)

        # Create root span (equivalent to the main method span)
        with tracer.start_as_current_span("virtual_query") as root_span:
            # Set root span attributes
            root_span.set_attribute(
                SpanAttributes.SPAN_TYPE,
                SpanAttributes.SpanType.RECORD_ROOT.value,
            )
            if input_value is not None:
                root_span.set_attribute(
                    SpanAttributes.RECORD_ROOT.INPUT, str(input_value)
                )
            if output_value is not None:
                root_span.set_attribute(
                    SpanAttributes.RECORD_ROOT.OUTPUT, str(output_value)
                )

            # Create retrieval span if contexts are available
            if contexts is not None:
                with tracer.start_as_current_span(
                    "virtual_retrieval"
                ) as retrieval_span:
                    retrieval_span.set_attribute(
                        SpanAttributes.SPAN_TYPE,
                        SpanAttributes.SpanType.RETRIEVAL.value,
                    )
                    if input_value is not None:
                        retrieval_span.set_attribute(
                            SpanAttributes.RETRIEVAL.QUERY_TEXT,
                            str(input_value),
                        )
                    retrieval_span.set_attribute(
                        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                        str(contexts),
                    )

            # Create generation span
            with tracer.start_as_current_span(
                "virtual_generation"
            ) as generation_span:
                generation_span.set_attribute(
                    SpanAttributes.SPAN_TYPE,
                    SpanAttributes.SpanType.GENERATION.value,
                )
                if output_value is not None:
                    generation_span.set_attribute(
                        SpanAttributes.GENERATION.OUTPUT, str(output_value)
                    )

    def _create_virtual_spans_without_app(
        self,
        input_id,
        input_value,
        output_value,
        contexts,
        input_records_count,
        ground_truth_output,
    ):
        """Create virtual spans without an app context by manually setting baggage and span attributes."""
        from opentelemetry import baggage
        from opentelemetry import trace
        import opentelemetry.context as context_api
        from trulens.experimental.otel_tracing.core.session import (
            TRULENS_SERVICE_NAME,
        )

        tracer = trace.get_tracer_provider().get_tracer(TRULENS_SERVICE_NAME)

        # Set baggage context manually for virtual runs
        ctx = baggage.set_baggage(ResourceAttributes.APP_NAME, self.object_name)
        ctx = baggage.set_baggage(
            ResourceAttributes.APP_VERSION, self.object_version or "1.0", ctx
        )
        ctx = baggage.set_baggage(SpanAttributes.RUN_NAME, self.run_name, ctx)
        ctx = baggage.set_baggage(SpanAttributes.INPUT_ID, input_id, ctx)
        ctx = baggage.set_baggage(
            SpanAttributes.INPUT_RECORDS_COUNT, str(input_records_count), ctx
        )
        if ground_truth_output:
            ctx = baggage.set_baggage(
                SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
                str(ground_truth_output),
                ctx,
            )

        token = context_api.attach(ctx)
        try:
            # Create root span (equivalent to the main method span)
            with tracer.start_as_current_span("virtual_query") as root_span:
                # Set root span attributes
                root_span.set_attribute(
                    SpanAttributes.SPAN_TYPE,
                    SpanAttributes.SpanType.RECORD_ROOT.value,
                )
                root_span.set_attribute(
                    ResourceAttributes.APP_NAME, self.object_name
                )
                root_span.set_attribute(
                    ResourceAttributes.APP_VERSION, self.object_version or "1.0"
                )
                root_span.set_attribute(SpanAttributes.RUN_NAME, self.run_name)
                root_span.set_attribute(SpanAttributes.INPUT_ID, input_id)
                root_span.set_attribute(
                    SpanAttributes.INPUT_RECORDS_COUNT, input_records_count
                )
                if input_value is not None:
                    root_span.set_attribute(
                        SpanAttributes.RECORD_ROOT.INPUT, str(input_value)
                    )
                if output_value is not None:
                    root_span.set_attribute(
                        SpanAttributes.RECORD_ROOT.OUTPUT, str(output_value)
                    )
                if ground_truth_output:
                    root_span.set_attribute(
                        SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
                        str(ground_truth_output),
                    )

                # Create retrieval span if contexts are available
                if contexts is not None:
                    with tracer.start_as_current_span(
                        "virtual_retrieval"
                    ) as retrieval_span:
                        retrieval_span.set_attribute(
                            SpanAttributes.SPAN_TYPE,
                            SpanAttributes.SpanType.RETRIEVAL.value,
                        )
                        if input_value is not None:
                            retrieval_span.set_attribute(
                                SpanAttributes.RETRIEVAL.QUERY_TEXT,
                                str(input_value),
                            )
                        retrieval_span.set_attribute(
                            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                            str(contexts),
                        )

                # Create generation span
                with tracer.start_as_current_span(
                    "virtual_generation"
                ) as generation_span:
                    generation_span.set_attribute(
                        SpanAttributes.SPAN_TYPE,
                        SpanAttributes.SpanType.GENERATION.value,
                    )
                    if output_value is not None:
                        generation_span.set_attribute(
                            SpanAttributes.GENERATION.OUTPUT, str(output_value)
                        )
        finally:
            context_api.detach(token)

    def _get_current_time_in_ms(self) -> int:
        return int(round(time.time() * 1000))

    def get_status(self) -> RunStatus:
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

        if run.run_status == "CANCELLED":
            return RunStatus.CANCELLED

        if not self._is_invocation_started(run):
            logger.info("Run is created, no invocation nor computation yet.")
            return RunStatus.CREATED
        elif self._metrics_computation_started(run):
            logger.info(
                "Run is created, invocation done, and some or all metrics computed."
            )

            return self._compute_overall_computations_status(run)

        else:
            logger.info("Run is created, invocation started.")

            return self._compute_latest_invocation_status(run)

    def _should_skip_computation(self, metric_name: str, run: Run) -> bool:
        metrics_metadata = (
            run.describe().get("run_metadata", {}).get("metrics", {})
        )

        if metrics_metadata is None:
            return False

        statuses = []  # will store statuses for all matching metric entries
        for metric_metadata in metrics_metadata.values():
            if metric_metadata.get("name", "") == metric_name:
                statuses.append(
                    metric_metadata.get("completion_status", {}).get("status")
                )

        # If no matching metric entries found, don't skip.
        if not statuses:
            return False

        if any(s == Run.CompletionStatusStatus.COMPLETED for s in statuses):
            logger.info(
                f"Metric {metric_name} already computed successfully (one entry COMPLETED); skipping computation."
            )
            return True

        # If any metric is in progress (i.e. started), we skip because it's still computing.
        if any(s == Run.CompletionStatusStatus.STARTED for s in statuses):
            logger.info(
                f"Metric {metric_name} is in progress (at least one entry not complete); skipping computation."
            )
            return True

        # If all matching metrics are FAILED, then allow re-computation.
        if all(s == Run.CompletionStatusStatus.FAILED for s in statuses):
            logger.info(
                f"All metric entries for {metric_name} have FAILED; allowing re-computation."
            )
            return False

        logger.warning(
            "Unknown state for metric computation; skipping computation."
        )
        return True

    def compute_metrics(self, metrics: List[str]) -> str:
        if not metrics:
            raise ValueError(
                "No metrics provided. Please provide at least one metric to compute."
            )

        run_status = self.get_status()

        logger.info(f"Current run status: {run_status}")
        if not self._can_start_new_metric_computation(run_status):
            return f"""Cannot start a new metric computation when in run status: {run_status}. Valid statuses are: {RunStatus.INVOCATION_COMPLETED}, {RunStatus.INVOCATION_PARTIALLY_COMPLETED},
        {RunStatus.COMPUTATION_IN_PROGRESS}, {RunStatus.COMPLETED}, {RunStatus.PARTIALLY_COMPLETED}, {RunStatus.FAILED}."""

        computed_metrics = []
        for metric in metrics:
            if self._should_skip_computation(metric, self):
                computed_metrics.append(metric)

        if computed_metrics:
            return (
                f"Cannot compute metrics because the following metric(s) are already computed or in progress: "
                f"{', '.join(computed_metrics)}. If you want to recompute, please cancel the run and start a new one."
            )

        logger.info(f"Metrics to compute: {metrics}.")

        self.run_dao.call_compute_metrics_query(
            metrics=metrics,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
            run_name=self.run_name,
        )

        logger.info("Metrics computation job started")
        return "Metrics computation in progress."

    def _is_cancelled(self) -> bool:
        return self.get_status() == RunStatus.CANCELLED

    def cancel(self):
        if self._is_cancelled():
            logger.warning(f"Run {self.run_name} is already cancelled.")
            return

        update_fields = {"run_status": "CANCELLED"}

        self.run_dao.upsert_run_metadata_fields(
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
            **update_fields,
        )

        logger.info(f"Run {self.run_name} cancelled.")

    def update(
        self, description: Optional[str] = None, label: Optional[str] = None
    ):
        """
        Only description and label are allowed to be updated at the moment.
        """
        update_fields = {}
        if description is not None:
            logger.info(f"Updating run description to {description}")
            update_fields["description"] = description
        if label is not None:
            logger.info(f"Updating run label to {label}")
            update_fields["labels"] = [label]

        if update_fields:
            self.run_dao.upsert_run_metadata_fields(
                run_name=self.run_name,
                object_name=self.object_name,
                object_type=self.object_type,
                object_version=self.object_version,
                **update_fields,
            )

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
