from __future__ import annotations  # defers evaluation of annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel
from pydantic import Field

DEFAULT_LLM_JUDGE_NAME = (
    "mistral-large2"  # TODO: finalize / modify after benchmarking is completed
)


class Run(BaseModel):
    class RunConfig(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        description: Optional[str] = Field(
            default=None, description="A description for the run."
        )
        label: Optional[str] = Field(
            default=None, description="A label categorizing the run."
        )
        dataset_fqn: Optional[str] = Field(
            default=None,
            description="The fully qualified name of the dataset (e.g. 'db.schema.user_table_name_1').",
        )

        dataset_col_spec: Optional[Dict[str, str]] = Field(
            default=None,
            description="Optional column name mapping from reserved dataset fields to column names in user's table.",
        )
        llm_judge_name: Optional[str] = Field(
            default=DEFAULT_LLM_JUDGE_NAME,
            description="Name of the LLM judge to be used for the run.",
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

    run_name: str = Field(..., description="Unique name of the run.")

    run_config: Optional[RunConfig] = Field(
        default=None,
        description="Run configuration that maintains states needed for app invocation and metrics computation.",
    )

    run_type: Optional[str] = Field(
        default=None, description="Type of the run. i.e. AI_EVALUATION"
    )

    description: Optional[str] = Field(
        default=None, description="Description of the run."
    )
    label: Optional[str] = Field(
        default=None, description="Label for grouping runs."
    )

    object_name: str = Field(
        ...,
        description="Name of the managing object (e.g. name of 'EXTERNAL_AGENT').",
    )

    object_type: str = Field(
        ..., description="Type of the managing object (e.g. 'EXTERNAL_AGENT')."
    )

    run_status: Optional[str] = Field(
        default=None, description="Status of the run."
    )  # should default be INACTIVE?

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"
        # allow custom obj like RunDao to be passed as a parameter and more importantly, account for
        # additional fields in Run metadata JSON response.

    def describe(self) -> Dict:
        """
        Retrieve the run metadata by querying the underlying DAO and return it as a dictionary.

        The underlying DAO method is expected to return
        a dictionary (JSON) representing the run's metadata response for flexibility (instead of a Run instance).
        """
        # TODO/TBD:  should we just return Run instance instead?

        result = self.run_dao.get_run(
            object_name=self.object_name, run_name=self.run_name
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
        )

    def start(self, batch_size: Optional[int] = None):
        # if self.run_config.input_df is not None:
        #     input_df = self.run_config.input_df
        # elif self.run_config.dataset_fqn is not None:
        #     dataset_fqn = self.run_config.dataset_fqn
        #     input_df = self.run_dao.session.sql(f"SELECT * FROM {dataset_fqn}").collect()

        # for row in input_df:
        #     input_id = input_df[col_spec .id_col]
        #     input = input_df[col_spec.input_cols]
        #     with custom_app as recording:
        # # bound main_method specified by user to test_app
        # return "run started, invocation done and ingestion in process"

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
