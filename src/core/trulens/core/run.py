from __future__ import annotations  # defers evaluation of annotations

import json
import logging
import re
import time
from typing import Any, ClassVar, Dict, List, Optional
import uuid

import pandas as pd
import pydantic
from pydantic import BaseModel
from pydantic import Field
from trulens.connectors.snowflake.dao.enums import CompletionStatusStatus

logger = logging.getLogger(__name__)


# Reserved fields (case-insensitive)
DATASET_RESERVED_FIELDS = {
    "input_id",  # Represents the unique identifier for the input.
    "input",  # Represents the main input column, allow optional subscripts like input_1, input_2, etc.
    "ground_truth_output",  # Represents the ground truth output, flexible in type (string or others)
}

INVOCATION_TIMEOUT_IN_MS = 3 * 60 * 1000  # 3 minutes in milliseconds


def validate_dataset_spec(
    dataset_spec: Dict[str, str],
) -> Dict[str, str]:
    """
    Validates and normalizes the dataset column specification to ensure it contains only
    valid fields and allows for subscripted fields like input_1, input_2, etc.

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

        # currently only handle subscripted 'input' columns, e.g., 'input_1', 'input_2'
        if "input" in normalized_key and normalized_key != "input_id":
            match = re.match(r"^input(?:_\d+)?$", normalized_key)
            if not match:
                raise ValueError(
                    f"Invalid subscripted input field '{key}'. Expected 'input' or 'input_1', 'input_2', etc."
                )

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

    class CompletionStatus(BaseModel):
        status: CompletionStatusStatus = Field(
            ..., description="The status of the completion."
        )
        record_count: Optional[int] = Field(
            default=None, description="The count of records processed."
        )

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

    def describe(self) -> Run:
        """
        Retrieve the metadata of the Run object.
        """
        # TODO/TBD:  should we just return Run instance instead?

        run_metadata_df = self.run_dao.get_run(
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
        )
        if run_metadata_df.empty:
            raise ValueError(f"Run {self.run_name} not found.")

        return Run.from_metadata_df(
            run_metadata_df,
            {
                "app": self,
                "main_method_name": self.main_method_name,
                "run_dao": self.run_dao,
                "tru_session": self.tru_session,
            },
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

    def start(self, input_df: Optional[pd.DataFrame] = None):
        """
        Start the run by invoking the main method of the user's app with the input data

        Args:
            input_df (Optional[pd.DataFrame], optional): user provided input dataframe.
        """

        if input_df is None:
            logger.info(
                "No input dataframe provided. Fetching input data from source."
            )
            # TODO: update the source_info.source_type to 'TABLE'
            rows = self.run_dao.session.sql(
                f"SELECT * FROM {self.source_info.name}"
            ).collect()
            input_df = pd.DataFrame([row.as_dict() for row in rows])

        dataset_column_spec = self.source_info.column_spec

        # Preprocess the dataset_column_spec to create mappings for input columns
        # and map the inputs for reserved fields only once, before the iteration over rows.
        input_columns_by_subscripts = {}
        reserved_field_column_mapping = {}

        # Process dataset column spec to handle subscripting logic for input columns
        for reserved_field, user_column in dataset_column_spec.items():
            if (
                reserved_field.startswith("input")
                or reserved_field.split("_")[1] == "input"
            ):
                if (
                    "_" in reserved_field
                    and reserved_field.split("_")[-1].isdigit()
                ):
                    subscript = int(reserved_field.split("_")[-1])
                    input_columns_by_subscripts[subscript] = user_column
                else:
                    input_columns_by_subscripts[0] = user_column
            else:
                # Prepare the kwargs for the non-input fields
                reserved_field_column_mapping[reserved_field] = user_column

        input_records_count = input_df.size

        invocation_metadata_id = self.run_dao._compute_invocation_metadata_id(
            dataset_name=self.source_info.name,
            input_records_count=input_records_count,
        )
        start_time_ms = int(round(time.time() * 1000))

        logger.info(
            f"Creating or updating invocation metadata with {input_records_count} records from input."
        )
        self.run_dao.upsert_invocation_metadata(
            invocation_metadata_id=invocation_metadata_id,
            input_records_count=input_records_count,
            start_time_ms=start_time_ms,
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
        )

        # user app invocation - will block until the app completes
        try:
            for i, row in input_df.iterrows():
                main_method_args = []

                # For each input column, add the value to main_method_args in the correct order
                for subscript in sorted(input_columns_by_subscripts.keys()):
                    user_column = input_columns_by_subscripts[subscript]
                    main_method_args.append(row[user_column])

                # Call the instrumented main method with the arguments
                input_id = (
                    row[dataset_column_spec["input_id"]]
                    if "input_id" in dataset_column_spec
                    else None
                )
                if input_id is None and "input" in dataset_column_spec:
                    input_id = hash(row[dataset_column_spec["input"]])

                self.app.instrumented_invoke_main_method(
                    run_name=self.run_name,
                    input_id=input_id,
                    main_method_args=tuple(
                        main_method_args
                    ),  # Ensure correct order
                    main_method_kwargs=None,  # don't take any kwargs for now so we don't break TruChain / TruLlama where input argument name cannot be defined by users.
                )
        except Exception as e:
            logger.exception(
                f"Error encountered during invoking app main method: {e}."
            )
            self.run_dao.upsert_invocation_metadata(
                invocation_metadata_id=invocation_metadata_id,
                end_time_ms=int(round(time.time() * 1000)),
                completion_status=Run.CompletionStatus(
                    status=CompletionStatusStatus.FAILED,
                ),
                run_name=self.run_name,
                object_name=self.object_name,
                object_type=self.object_type,
                object_version=self.object_version,
            )

            raise

        self.tru_session.force_flush()
        logger.info("Run started, invocation done and ingestion in process.")

    def _read_record_count_from_event_table(self) -> int:
        # TODO: check w/ Dave to fix this query
        q = """
            SELECT
                *
            FROM
                table(snowflake.local.GET_AI_OBSERVABILITY_EVENTS(
                    ?,
                    ?,
                    ?,
                    'EXTERNAL AGENT'
                ))
            WHERE
                RECORD_ATTRIBUTES:"snow.ai.observability.run.name" = ?
            """
        try:
            ret = self.run_dao.session.sql(
                q,
                params=[
                    self.run_dao.session.get_current_database()[1:-1],
                    self.run_dao.session.get_current_schema()[1:-1],
                    self.object_name,
                    self.run_name,
                ],
            ).to_pandas()

            return len(ret)
        except Exception as e:
            logger.exception(
                f"Error encountered during reading record count from event table: {e}."
            )
            raise

    def get_status(self) -> str:
        # first read from DPO backend, decide if we need to update status, and return
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

        if not run.run_metadata.invocations:
            return "CREATED"
        elif run.run_metadata.invocations and not run.run_metadata.computations:
            latest_invocation = max(
                run.run_metadata.invocations.values(),
                key=lambda inv: inv.start_time_ms or 0,
            )
            logger.info(f"latest invocation field  {latest_invocation}")

            if (
                latest_invocation.completion_status
                and latest_invocation.completion_status.status
            ):
                return (
                    "INVOCATION_" + latest_invocation.completion_status.status
                )

            current_ingested_records_count = (
                self._read_record_count_from_event_table()
            )
            logger.info(
                f"Current ingested records count: {current_ingested_records_count}"
            )
            if (
                latest_invocation.start_time_ms
                and time.time() * 1000 - latest_invocation.start_time_ms
                > INVOCATION_TIMEOUT_IN_MS
            ):
                logger.warning("Invocation timeout reached and concluded")
                # timeout reached, persist to DPO backend
                self.run_dao.upsert_invocation_metadata(
                    invocation_metadata_id=latest_invocation.id,
                    end_time_ms=int(round(time.time() * 1000)),
                    completion_status=Run.CompletionStatus(
                        status=CompletionStatusStatus.PARTIALLY_COMPLETED,
                        record_count=current_ingested_records_count,
                    ),
                    run_name=self.run_name,
                    object_name=self.object_name,
                    object_type=self.object_type,
                    object_version=self.object_version,
                )
                return "INVOCATION_PARTIALLY_COMPLETED"

            elif (
                latest_invocation.input_records_count
                and current_ingested_records_count
                >= latest_invocation.input_records_count
            ):
                # happy case, add end time and update status
                self.run_dao.upsert_invocation_metadata(
                    invocation_metadata_id=latest_invocation.id,
                    end_time_ms=int(round(time.time() * 1000)),
                    completion_status=Run.CompletionStatus(
                        status=CompletionStatusStatus.COMPLETED,
                        record_count=current_ingested_records_count,
                    ).model_dump(),
                    run_name=self.run_name,
                    object_name=self.object_name,
                    object_type=self.object_type,
                    object_version=self.object_version,
                )
                return "INVOCATION_COMPLETED"
            elif latest_invocation.end_time_ms == 0:
                return "INVOCATION_IN_PROGRESS"

        elif run.run_metadata.computations:
            return "SOME COMPUTATION STATUS"
        else:
            return "UNKNOWN"

    def compute_metrics(self, metrics: List[str]):
        # TODO: add update operations to the run metadata
        run_status = self.get_status()
        logger.info(f"Current run status: {run_status}")
        if (
            run_status == "INVOCATION_COMPLETED"
            or run_status == "INVOCATION_PARTIALLY_COMPLETED"
        ):
            current_db = self.run_dao.session.get_current_database()
            current_schema = self.run_dao.session.get_current_schema()
            if not metrics:
                raise ValueError("Metrics list cannot be empty")
            metrics_str = ",".join([f"'{metric}'" for metric in metrics])
            compute_metrics_query = f"CALL COMPUTE_AI_OBSERVABILITY_METRICS('{current_db}', '{current_schema}', '{self.object_name}', '{self.object_version}', '{self.object_type}', '{self.run_name}', ARRAY_CONSTRUCT({metrics_str}));"

            compute_query = self.run_dao.session.sql(compute_metrics_query)

            async_job = compute_query.collect_nowait()
            query_id = async_job.query_id
            logger.info(f"Query id for metrics computation: {query_id}")
            computation_metadata_id = str(uuid.uuid4())
            self.run_dao.upsert_computation_metadata(
                computation_metadata_id=computation_metadata_id,
                query_id=query_id,
                run_name=self.run_name,
                object_name=self.object_name,
                object_type=self.object_type,
                object_version=self.object_version,
                start_time_ms=int(round(time.time() * 1000)),
            )

            compute_results_rows = async_job.result()

            for row in compute_results_rows:
                row_msg = row["MESSAGE"]
                logger.error(row_msg)
                # computed_records_count = int(
                #     row_msg.split(" ")[-1]
                # )  # TODO change to regex or directly read the field when available

                # if row["STATUS"] == "SUCCESS":
                #     logger.info(
                #         f"Metrics computation for {row['METRIC']} succeeded."
                #     )
                #     self.run_dao.upsert_metrics_metadata(
                #         metrics_metadata_id="XXX",
                #         computation_id=computation_metadata_id,
                #         name=row["METRIC"],
                #         completion_status=Run.CompletionStatus(
                #             status=CompletionStatusStatus.COMPLETED,
                #             record_count=computed_records_count,
                #         ).model_dump(),
                #         run_name=self.run_name,
                #         object_name=self.object_name,
                #         object_type=self.object_type,
                #         object_version=self.object_version,
                #     )

            return
        else:
            return f"Cannot start metrics computation yet when in run status: {run_status}"

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
