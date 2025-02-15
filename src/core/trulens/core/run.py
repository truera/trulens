from __future__ import annotations  # defers evaluation of annotations

import json
from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd
import pydantic
from pydantic import BaseModel
from pydantic import Field


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
        ..., description="App instance to be invoked during run.", exclude=True
    )

    main_method_name: str = Field(
        ..., description="Main method of the app.", exclude=True
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
        llmJudgeName: Optional[str] = (
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
        raise NotImplementedError("start is not implemented yet.")

    def get_status(self):
        raise NotImplementedError("status is not implemented yet.")

    def compute_metrics(self, metrics: List[str], params: dict):
        raise NotImplementedError("compute_metrics is not implemented yet.")

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
