from __future__ import annotations  # defers evaluation of annotations

from enum import Enum
import inspect
import json
import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Set, Type, Union

import pandas as pd
import pydantic
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_serializer
from trulens.core.enums import Mode
from trulens.core.feedback.custom_metric import MetricConfig
from trulens.core.metric import metric as metric_module
from trulens.core.utils.json import obj_id_of_obj
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


def _parse_json_field(
    value: Any,
    field_name: str,
    index: Optional[int] = None,
    critical: bool = True,
) -> Any:
    """
    Helper function to parse JSON fields consistently.

    Args:
        value: The value to parse (could be string or already parsed dict)
        field_name: Name of the field being parsed (for logging)
        index: Optional index for logging context
        critical: Whether parsing failure should cause the caller to continue processing

    Returns:
        Parsed dictionary or the original value if already parsed

    Raises:
        Continues processing on JSONDecodeError if critical=False, otherwise logs warning
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            index_msg = f" at index {index}" if index is not None else ""
            if critical:
                logger.warning(f"Failed to parse {field_name} JSON{index_msg}")
                raise
            else:
                logger.debug(f"Failed to parse {field_name} JSON{index_msg}")
                return value
    return value


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
    mode: Mode = Field(
        default=Mode.APP_INVOCATION,
        description="Mode of operation: LOG_INGESTION for creating spans from existing data, APP_INVOCATION for instrumenting spans from a new app execution.",
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

    main_method_name: Optional[str] = Field(
        default=None, description="Main method of the app.", exclude=True
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
        mode: Optional[str] = Field(
            default=Mode.APP_INVOCATION.value,
            description="Mode of operation: LOG_INGESTION or APP_INVOCATION.",
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
            rows = self.run_dao.session.sql(
                f"SELECT * FROM {self.source_info.name}"
            ).collect()
            input_df = pd.DataFrame([row.as_dict() for row in rows])

        dataset_spec = self.source_info.column_spec

        input_records_count = len(input_df)

        logger.info(
            f"Creating or updating invocation metadata with {input_records_count} records from input."
        )

        # Determine mode from run metadata
        mode = self.run_metadata.mode or Mode.APP_INVOCATION.value

        try:
            if mode == Mode.LOG_INGESTION.value:
                self._create_virtual_spans(
                    input_df, dataset_spec, input_records_count
                )
            else:
                # user app invocation - will block until the app completes
                for _, row in input_df.iterrows():
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
                            input_value = row[input_col]
                            # Parse JSON string to dict if the input is a JSON blob
                            # This supports LangGraph state dicts from Snowflake VARIANT columns
                            if isinstance(input_value, str):
                                try:
                                    import json

                                    parsed = json.loads(input_value)
                                    if isinstance(parsed, dict):
                                        input_value = parsed
                                except (json.JSONDecodeError, TypeError):
                                    pass  # Keep as string if not valid JSON
                            input_id = obj_id_of_obj(input_value)
                            main_method_args.append(input_value)

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
        This method creates spans dynamically based on the dataset_spec mapping,
        which maps span attribute paths to column names.

        The input DataFrame must conform to a specific format based on the dataset_spec.
        Required columns/data:
        - If 'input_id' is specified in dataset_spec: A column containing unique identifiers
        - If no 'input_id': Either 'input' or 'record_root.input' column must exist to generate IDs
        - If neither exists, row indices will be used as fallback IDs

        Optional columns (specified via dataset_spec):
        - 'ground_truth_output' or 'record_root.ground_truth_output': Ground truth data
        - Any additional columns mapped in dataset_spec will be used to populate span attributes

        See RECORD_ROOT and other span attribute definitions in trulens.otel.semconv.trace
        for the full set of supported attribute paths that can be mapped to columns.
        """

        logger.debug(f"Creating virtual spans for {len(input_df)} records")

        for i, row in input_df.iterrows():
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

            ground_truth_output = row.get(
                dataset_spec.get("ground_truth_output")
                or dataset_spec.get("record_root.ground_truth_output")
            )

            # Ensure input_id is never None - use row index as fallback
            if input_id is None:
                input_id = f"row_{i}"
                logger.warning(f"input_id was None, using fallback: {input_id}")

            try:
                self._create_virtual_spans_with_nested_contexts(
                    row,
                    dataset_spec,
                    input_id,
                    input_records_count,
                    ground_truth_output,
                )
            except Exception as e:
                logger.exception(f"Error in virtual span creation: {e}")
                raise

    def _create_virtual_spans_with_nested_contexts(
        self,
        row,
        dataset_spec,
        input_id,
        input_records_count,
        ground_truth_output,
    ):
        """Create virtual spans using OtelRecordingContext - simplified approach"""
        from trulens.core.otel.instrument import OtelRecordingContext

        with OtelRecordingContext(
            tru_app=self.app,
            app_name=self.app.app_name,
            app_version=self.app.app_version,
            run_name=self.run_name,
            input_id=input_id,
            input_records_count=input_records_count,
            ground_truth_output=ground_truth_output,
        ):
            # Now create nested spans within this context
            self._create_nested_spans_from_dataset_spec(dataset_spec, row)

            logger.debug("Created all nested spans")

    def _create_nested_spans_from_dataset_spec(
        self, dataset_spec: Dict[str, str], row
    ):
        """Create properly nested spans within OtelRecordingContext"""
        from trulens.experimental.otel_tracing.core.span import (
            set_general_span_attributes,
        )

        span_data = self._group_dataset_spec_by_span_type(dataset_spec, row)
        logger.info(f"Creating nested spans for: {list(span_data.keys())}")

        if "record_root" in span_data:
            from trulens.core.otel.function_call_context_manager import (
                create_function_call_context_manager,
            )

            with create_function_call_context_manager(
                True, "virtual_record"
            ) as root_span:
                set_general_span_attributes(
                    root_span, SpanAttributes.SpanType.RECORD_ROOT
                )
                self._set_span_attributes_from_data(
                    root_span, span_data["record_root"], "record_root"
                )

                root_span_id = root_span.get_span_context().span_id
                logger.debug(
                    f"Created record_root span with ID: {root_span_id}"
                )

                # Create child spans nested WITHIN the root span context
                for span_type, attributes in span_data.items():
                    if span_type == "record_root":
                        continue

                    span_type_enum = self._get_span_type_enum(span_type)
                    if span_type_enum:
                        logger.debug(
                            f"Creating child span: {span_type} under parent {root_span_id}"
                        )
                        with create_function_call_context_manager(
                            True, f"{span_type}_virtual"
                        ) as child_span:
                            set_general_span_attributes(
                                child_span, span_type_enum
                            )
                            self._set_span_attributes_from_data(
                                child_span, attributes, span_type
                            )

                            child_span_id = (
                                child_span.get_span_context().span_id
                            )
                            child_parent_span_id = (
                                child_span.parent.span_id
                                if child_span.parent
                                else "None"
                            )
                            logger.debug(
                                f"Created {span_type} child span ID: {child_span_id}, parent ID: {child_parent_span_id}"
                            )
                    else:
                        logger.warning(
                            f"Unknown span type: {span_type} - skipping"
                        )
        else:
            logger.warning("No record_root defined in dataset_spec")

    def _set_span_attributes_from_data(
        self, span, attributes: Dict[str, any], span_type: str
    ):
        """Set span attributes using proper SpanAttributes constants"""
        span_attrs_class = self._get_span_attributes_class(span_type)

        for attr_name, value in attributes.items():
            if span_attrs_class:
                attr_constant = getattr(
                    span_attrs_class, attr_name.upper(), None
                )
                if attr_constant:
                    # Check if this attribute should be treated as an array
                    if self._should_process_as_array(attr_constant):
                        processed_value = self._process_array_attribute(value)
                        span.set_attribute(attr_constant, processed_value)
                        logger.info(
                            f"Set {span_type} attribute {attr_constant} = {processed_value}"
                        )
                    else:
                        span.set_attribute(attr_constant, str(value))
                        logger.info(
                            f"Set {span_type} attribute {attr_constant} = {value}"
                        )
                else:
                    # Fallback to raw attribute name
                    fallback_key = f"{span_type}.{attr_name}"
                    # Check if this attribute name suggests it should be an array
                    if self._should_process_as_array_by_name(attr_name):
                        processed_value = self._process_array_attribute(value)
                        span.set_attribute(fallback_key, processed_value)
                        logger.info(
                            f"Set fallback attribute {fallback_key} = {processed_value}"
                        )
                    else:
                        span.set_attribute(fallback_key, str(value))
                        logger.info(
                            f"Set fallback attribute {fallback_key} = {value}"
                        )
            else:
                fallback_key = f"{span_type}.{attr_name}"
                # Check if this attribute name suggests it should be an array
                if self._should_process_as_array_by_name(attr_name):
                    processed_value = self._process_array_attribute(value)
                    span.set_attribute(fallback_key, processed_value)
                    logger.info(
                        f"Set fallback attribute {fallback_key} = {processed_value}"
                    )
                else:
                    span.set_attribute(fallback_key, str(value))
                    logger.info(
                        f"Set fallback attribute {fallback_key} = {value}"
                    )

    def _should_process_as_array(self, attr_constant: str) -> bool:
        """
        Determine if a span attribute constant should be processed as an array.

        Args:
            attr_constant: The span attribute constant (e.g., SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS)

        Returns:
            bool: True if the attribute should be processed as an array
        """
        # Define attributes that should be treated as arrays based on semantic conventions
        array_attributes = {
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            SpanAttributes.GRAPH_NODE.NODES_EXECUTED,
            SpanAttributes.RERANKER.INPUT_CONTEXT_TEXTS,
            SpanAttributes.RERANKER.INPUT_CONTEXT_SCORES,
            SpanAttributes.RERANKER.INPUT_RANKS,
            SpanAttributes.RERANKER.OUTPUT_RANKS,
            SpanAttributes.RERANKER.OUTPUT_CONTEXT_TEXTS,
            SpanAttributes.RERANKER.OUTPUT_CONTEXT_SCORES,
        }
        return attr_constant in array_attributes

    def _should_process_as_array_by_name(self, attr_name: str) -> bool:
        """
        Determine if an attribute should be processed as an array based on its name.
        This is used for fallback cases where we don't have the constant.

        Args:
            attr_name: The attribute name (e.g., "retrieved_contexts")

        Returns:
            bool: True if the attribute should be processed as an array
        """
        # Attribute names that suggest array content
        array_attribute_names = {
            "retrieved_contexts",
            "nodes_executed",
            "input_context_texts",
            "input_context_scores",
            "input_ranks",
            "output_ranks",
            "output_context_texts",
            "output_context_scores",
        }
        return attr_name.lower() in array_attribute_names

    def _process_array_attribute(self, value: any) -> List[str]:
        """
        Process a value that should be treated as an array attribute.

        Args:
            value: The raw value from the dataset, could be a string, list, or other type

        Returns:
            List[str]: A list of string values
        """
        if value is None:
            return []

        # If it's already a list, return it (converting items to strings)
        if isinstance(value, list):
            return [str(item) for item in value]

        # If it's a string, try to parse it as JSON first, then fall back to comma-separated
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []

            # Try to parse as JSON array first
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

            # Fall back to comma-separated parsing
            # Split by comma and clean up whitespace
            items = [item.strip() for item in value.split(",")]
            # Remove empty strings
            items = [item for item in items if item]
            return items

        # For any other type, convert to string and treat as single item
        return [str(value)]

    def _group_dataset_spec_by_span_type(
        self, dataset_spec: Dict[str, str], row
    ) -> Dict[str, Dict[str, any]]:
        """Group dataset_spec entries by span type (record_root, retrieval, generation, etc.)"""
        span_data = {}

        for spec_key, column_name in dataset_spec.items():
            if column_name not in row:
                continue

            value = row[column_name]
            parts = spec_key.lower().split(".")

            if len(parts) >= 2:
                span_type = f"{parts[0]}"  # e.g., 'record_root'
                attribute = ".".join(
                    parts[1:]
                )  # e.g., 'input', 'output', 'query_text'
            else:
                logger.warning(
                    f"Legacy key: {spec_key} - assuming it's record_root "
                )
                span_type = "record_root"
                attribute = parts[0]

            if span_type not in span_data:
                span_data[span_type] = {}

            span_data[span_type][attribute] = value

        return span_data

    def _get_span_type_enum(self, span_type: str):
        """Map span type string to SpanType enum dynamically"""
        # Convert span_type to uppercase to match enum naming convention
        enum_name = span_type.upper()

        # Try to get the enum value dynamically
        return getattr(SpanAttributes.SpanType, enum_name, None)

    def _get_span_attributes_class(self, span_type: str):
        """Get the appropriate SpanAttributes class for a span type dynamically"""
        # Convert span_type to uppercase to match class naming convention
        class_name = span_type.upper()

        # Try to get the attributes class dynamically
        return getattr(SpanAttributes, class_name, None)

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

    def compute_metrics(self, metrics: List[Union[str, MetricConfig]]) -> str:
        """
        Compute metrics for the run.

        Args:
            metrics: List of metric identifiers (strings) for server-side metrics,
                    or MetricConfig objects for client-side metrics

        Returns:
            Status message indicating computation progress
        """
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

        # Separate client-side and server-side metrics
        client_metric_configs = []
        server_metric_names = []

        for metric in metrics:
            if isinstance(metric, str):
                # String metrics are server-side
                server_metric_names.append(metric)
            else:
                # MetricConfig objects are client-side
                if (
                    hasattr(metric, "computation_type")
                    and metric.computation_type == "client"
                ):
                    client_metric_configs.append(metric)
                else:
                    # Default to client-side for MetricConfig objects
                    client_metric_configs.append(metric)

        logger.info(
            f"Client-side metrics to compute: {[m.name for m in client_metric_configs]}"
        )
        logger.info(f"Server-side metrics to compute: {server_metric_names}")

        try:
            # Handle client-side metrics
            if client_metric_configs:
                try:
                    self._compute_client_side_metrics_from_configs(
                        client_metric_configs
                    )
                    logger.info(
                        f"Successfully computed {len(client_metric_configs)} client-side metrics"
                    )
                except Exception as e:
                    logger.error(f"Error computing client-side metrics: {e}")
                    raise

            if server_metric_names:
                self.run_dao.call_compute_metrics_query(
                    metrics=server_metric_names,
                    object_name=self.object_name,
                    object_type=self.object_type,
                    object_version=self.object_version,
                    run_name=self.run_name,
                )
                logger.info(
                    f"Started server-side computation for {len(server_metric_names)} metrics"
                )

            logger.info("Metrics computation job started")
        finally:
            self.tru_session.force_flush()

            logger.debug(
                "Flushed OTel eval spans to event table to ensure all spans are ingested before main process exits"
            )

        return "Metrics computation in progress."

    def _compute_client_side_metrics_from_configs(
        self,
        metric_configs: List[Union[MetricConfig, "metric_module.Metric"]],
    ) -> None:
        """Compute client-side custom metrics from Metric or MetricConfig objects."""
        try:
            from trulens.feedback.computer import compute_feedback_by_span_group
        except ImportError:
            logger.error(
                "trulens.feedback package is not installed. Please install it to use feedback computation functionality."
            )
            raise

        # Force flush to ensure spans are uploaded to Snowflake before querying
        self.tru_session.force_flush()
        logger.debug(
            "Flushed OTel spans to event table before retrieving them for client-side metric computation"
        )

        events = self._get_events_for_client_metrics()

        if events.empty:
            logger.warning(
                f"No events found for app {self.app.app_name} version {self.app.app_version} run {self.run_name}"
            )
            return

        # Compute each client-side metric

        for metric_config in metric_configs:
            try:
                logger.info(
                    f"Computing client-side metric: {metric_config.name}"
                )

                # Handle both Metric objects (new API) and MetricConfig (deprecated)
                if isinstance(metric_config, metric_module.Metric):
                    # Metric objects are already feedback definitions
                    feedback = metric_config
                elif hasattr(metric_config, "create_feedback_definition"):
                    # MetricConfig (deprecated) needs conversion
                    feedback = metric_config.create_feedback_definition()
                else:
                    raise TypeError(
                        f"Expected Metric or MetricConfig, got {type(metric_config)}"
                    )

                compute_feedback_by_span_group(
                    events=events,
                    feedback=feedback,
                    raise_error_on_no_feedbacks_computed=False,
                    selectors=feedback.selectors,
                )
                logger.info(
                    f"Successfully computed client-side metric: {metric_config.name}"
                )
            except Exception as e:
                logger.error(
                    f"Error computing client-side metric {metric_config.name}: {e}"
                )
                raise

    def _get_events_for_client_metrics(self) -> pd.DataFrame:
        """Get events for client-side metric computation using the appropriate method."""
        try:
            from trulens.connectors.snowflake import SnowflakeConnector

            if (
                isinstance(self.tru_session.connector, SnowflakeConnector)
                and self.tru_session.connector.use_account_event_table
            ):
                events_df = self.tru_session.connector.db.get_events(
                    app_name=self.app.app_name,
                    app_version=self.app.app_version,
                    run_name=self.run_name,
                )

                if not events_df.empty:
                    for json_col in [
                        "TRACE",
                        "RESOURCE_ATTRIBUTES",
                        "RECORD",
                        "RECORD_ATTRIBUTES",
                    ]:
                        if json_col in events_df.columns:
                            events_df[json_col] = events_df[json_col].apply(
                                json.loads
                            )

                    # Rename columns to match expected format (lowercase) in downstream feedback computation
                    # TODO: remove this once we have a more robust/general way to handle the column names
                    column_mapping = {
                        "TRACE": "trace",
                        "RESOURCE_ATTRIBUTES": "resource_attributes",
                        "RECORD": "record",
                        "RECORD_ATTRIBUTES": "record_attributes",
                    }

                    events_df = events_df.rename(columns=column_mapping)

                    for idx, row in events_df.iterrows():
                        trace = events_df.at[idx, "trace"]
                        record = events_df.at[idx, "record"]
                        record_attributes = (
                            events_df.at[idx, "record_attributes"]
                            if "record_attributes" in events_df.columns
                            else {}
                        )

                        # Parse JSON fields using helper function
                        try:
                            trace = _parse_json_field(
                                trace, "trace", idx, critical=True
                            )
                            events_df.at[idx, "trace"] = trace
                        except json.JSONDecodeError:
                            continue

                        try:
                            record = _parse_json_field(
                                record, "record", idx, critical=True
                            )
                            events_df.at[idx, "record"] = record
                        except json.JSONDecodeError:
                            continue

                        # record_attributes parsing is not critical for parent_id assignment
                        record_attributes = _parse_json_field(
                            record_attributes,
                            "record_attributes",
                            idx,
                            critical=False,
                        )
                        events_df.at[idx, "record_attributes"] = (
                            record_attributes
                        )

                        # Now modify the dictionary
                        if isinstance(trace, dict) and isinstance(record, dict):
                            if "parent_span_id" in record:
                                trace["parent_id"] = record["parent_span_id"]
                            else:
                                trace["parent_id"] = None

                            # Set the modified dict back into the DataFrame
                            events_df.at[idx, "trace"] = trace

                if not events_df.empty and self.run_name:
                    filtered_events = []
                    for _, row in events_df.iterrows():
                        try:
                            record_attributes = row.get("record_attributes", {})

                            # Parse record_attributes using helper function
                            original_record_attributes = record_attributes
                            record_attributes = _parse_json_field(
                                record_attributes,
                                "record_attributes",
                                critical=False,
                            )
                            # If parsing failed and we still have a string, skip this record
                            if (
                                isinstance(record_attributes, str)
                                and record_attributes
                                == original_record_attributes
                            ):
                                continue

                            event_run_name = record_attributes.get(
                                SpanAttributes.RUN_NAME
                            )
                            if event_run_name == self.run_name:
                                filtered_events.append(row)

                        except Exception as e:
                            logger.debug(
                                f"Skipping event due to parsing error: {e}"
                            )
                            continue

                    if filtered_events:
                        events_df = pd.DataFrame(filtered_events)
                        logger.info(
                            f"Filtered {len(filtered_events)} events for run {self.run_name}"
                        )
                    else:
                        logger.warning(
                            f"No events found for run {self.run_name} after filtering"
                        )
                        events_df = pd.DataFrame()

                return events_df
            else:
                return self.tru_session.connector.get_events(
                    app_name=self.app.app_name,
                    app_version=self.app.app_version,
                    run_name=self.run_name,
                )
        except ImportError:
            raise ValueError(
                "Snowflake connector is not installed. Please install it to use feedback computation functionality."
            )

    def get_records(
        self,
        record_ids: Optional[List[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        A wrapper API around get_records_and_feedback to retrieve and display overview of records from event table of the run.
        It aggregates summary information of records into a single DataFrame.

        Args:
            record_ids: Optional list of record IDs to filter by. Defaults to None.
            offset: Record row offset.
            limit: Limit on the number of records to return.

        Returns:
            A DataFrame with the overview of records.
        """
        record_details_df, metrics_columns = (
            self.tru_session.get_records_and_feedback(
                app_name=self.object_name,
                app_version=self.object_version,
                run_name=self.run_name,
                record_ids=record_ids,
                offset=offset,
                limit=limit,
            )
        )

        record_overview_col_names = [
            "record_id",
            "input",
            "output",
            "latency",
        ] + metrics_columns
        return record_details_df[record_overview_col_names]

    def get_record_details(
        self,
        record_ids: Optional[List[str]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        A wrapper API around get_records_and_feedback to retrieve records from event table of the run.

        Args:
            record_ids: Optional list of record IDs to filter by. Defaults to None.
            offset: Record row offset.
            limit: Limit on the number of records to return.

        Returns:
            A DataFrame with the details of records.
        """
        record_details_df, _ = self.tru_session.get_records_and_feedback(
            app_name=self.object_name,
            app_version=self.object_version,
            run_name=self.run_name,
            record_ids=record_ids,
            offset=offset,
            limit=limit,
        )

        return record_details_df

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
