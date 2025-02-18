from __future__ import annotations  # defers evaluation of annotations

import json
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd
import pydantic
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


# Reserved fields (case-insensitive)
DATASET_RESERVED_FIELDS = {
    "input_id",  # Represents the unique identifier for the input.
    "input",  # Represents the main input column, allow optional subscripts like input_1, input_2, etc.
    "ground_truth_output",  # Represents the ground truth output, flexible in type (string or others)
}


def validate_dataset_col_spec(
    dataset_col_spec: Dict[str, str],
) -> Dict[str, str]:
    """
    Validates and normalizes the dataset column specification to ensure it contains only
    valid fields and allows for subscripted fields like input_1, input_2, etc.

    Args:
        dataset_col_spec: The user-provided dictionary with column names.

    Returns:
        A validated and normalized dictionary.

    Raises:
        ValueError: If any invalid field is present.
    """

    normalized_spec = {}

    for key, value in dataset_col_spec.items():
        normalized_key = key.lower()

        # Ensure that the key is one of the valid reserved fields or its subscripted form
        if not any(
            normalized_key.startswith(reserved_field)
            for reserved_field in DATASET_RESERVED_FIELDS
        ):
            raise ValueError(
                f"Invalid field '{key}' found in dataset_col_spec."
            )

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
        description="Mandatory field. The fully qualified name of a user's Table / View  (e.g. 'db.schema.user_table_name_1'), or any user specified name of input dataframe.",
    )

    dataset_col_spec: Dict[str, str] = Field(
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

    class RunMetada(BaseModel):
        description: Optional[str] = Field(
            default=None, description="A description for the run."
        )
        labels: List[Optional[str]] = Field(
            default=[],
            description="Text label to group the runs. Take a single label for now",
        )
        llm_judge_name: Optional[str] = (
            Field(  # TODO: daniel - this needs to be udpated to `llm_judge_name`
                default=None,
                description="Name of the LLM judge to be used for the run.",
            )
        )

    run_metadata: RunMetada = Field(
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
            default="TABLE",
            description="Type of the source (e.g. 'TABLE').",
        )

    source_info: SourceInfo = Field(
        default=...,
        description="Source information for the run.",
    )

    def describe(self) -> Dict:
        """
        Retrieve the run metadata by querying the underlying DAO and return it as a dictionary.

        The underlying DAO method is expected to return
        a dictionary (JSON) representing the run's metadata response for flexibility (instead of a Run instance).
        """
        # TODO/TBD:  should we just return Run instance instead?

        result = self.run_dao.get_run(
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
            object_version=self.object_version,
        )
        if isinstance(result, dict):
            return result
        elif hasattr(result, "empty") and not result.empty:
            # Return the first row as a dictionary.
            return result.iloc[0].to_dict()
        else:
            return {}

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
        # TODO: add update operations to the run metadata
        if input_df is None:
            logger.info(
                "No input dataframe provided. Fetching input data from source."
            )
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

        for _, row in input_df.iterrows():
            main_method_args = []

            # For each input column, add the value to main_method_args in the correct order
            for subscript in sorted(input_columns_by_subscripts.keys()):
                user_column = input_columns_by_subscripts[subscript]
                main_method_args.append(row[user_column])

            # Ensure that main_method_kwargs uses the correct column values from the row
            main_method_kwargs = {
                key: row[value]
                for key, value in reserved_field_column_mapping.items()
                if value in row
            }

            # Call the instrumented main method with the arguments
            self.app.instrumented_invoke_main_method(
                run_name=self.run_name,
                input_id=row[dataset_column_spec["input_id"]]
                if "input_id" in dataset_column_spec
                else None,
                main_method_args=tuple(
                    main_method_args
                ),  # Ensure correct order
                main_method_kwargs=main_method_kwargs,  # Include only relevant kwargs
            )

        self.tru_session.force_flush()
        logger.info("Run started, invocation done and ingestion in process.")

    def get_status(self):
        raise NotImplementedError("status is not implemented yet.")

    def compute_metrics(self, metrics: List[str]):
        # TODO: add update operations to the run metadata

        current_db = self.run_dao.session.get_current_database()
        current_schema = self.run_dao.session.get_current_schema()
        if not metrics:
            raise ValueError("Metrics list cannot be empty")
        metrics_str = ",".join([f"'{metric}'" for metric in metrics])
        compute_metrics_query = f"CALL COMPUTE_AI_OBSERVABILITY_METRICS('{current_db}', '{current_schema}', '{self.object_name}', '{self.object_type}', '{self.run_name}', ARRAY_CONSTRUCT({metrics_str}));"

        return self.run_dao.session.sql(compute_metrics_query).collect()

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
