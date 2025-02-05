from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from trulens.connectors.snowflake.dao.run import RunDao


class RunConfig(BaseModel):
    description: Optional[str] = Field(
        None, description="A description for the run."
    )
    label: Optional[str] = Field(
        None, description="A label categorizing the run."
    )
    dataset_fqn: Optional[str] = Field(
        None,
        description="The fully qualified name of the dataset (e.g. 'db.schema.user_table_name_1').",
    )
    input_df: Optional[pd.DataFrame] = Field(
        None,
        description="The input dataset as a pandas DataFrame.",
    )

    @model_validator(mode="before")
    def check_run_data_source(cls, values: dict) -> dict:
        dataset_fqn = values.get("dataset_fqn")
        input_df = values.get("input_df")
        # Ensure that exactly one of dataset_fqn and input_df is provided.
        if (dataset_fqn is None and input_df is None) or (
            dataset_fqn is not None and input_df is not None
        ):
            raise ValueError(
                "Either 'dataset_fqn' or 'input_df' must be provided, but not both."
            )
        return values

    dataset_col_spec: Optional[Dict[str, str]] = Field(
        None,
        description="Optional column name mapping from reserved dataset fields to column names in user's table.",
    )
    llm_judge_name: Optional[str] = Field(
        "mistral-large2",  # TODO: PENDING, to be finalized after judge benchmarking is done
        description="The name of the LLM judge to use (e.g. 'mistral-large2').",
    )


class Run(BaseModel):
    """
    Run class for managing run state / attributes in the SDK client.

    This model is meant to be used and accessed through
    methods like describe() (which uses the underlying RunDao) to obtain the run metadata.
    """

    _run_dao: RunDao = Field(
        ..., description="DAO instance for run operations.", exclude=True
    )

    _app: Any = Field(
        ..., description="App instance to be invoked during run.", exclude=True
    )

    _main_method_name: str = Field(
        ..., description="Main method of the app.", exclude=True
    )

    run_name: str = Field(..., description="Unique name of the run.")

    run_config: RunConfig = Field(
        ...,
        description="Run configuration that maintains states needed for app invocation and metrics computation.",
    )

    description: Optional[str] = Field(
        None, description="Description of the run."
    )
    label: Optional[str] = Field(None, description="Label for grouping runs.")

    object_name: str = Field(
        ...,
        description="Name of the managing object (e.g. name of 'EXTERNAL_AGENT').",
    )

    object_type: str = Field(
        ..., description="Type of the managing object (e.g. 'EXTERNAL_AGENT')."
    )

    status: Optional[str] = Field(None, description="Status of the run.")

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"
        # allow custom obj like RunDao to be passed as a parameter and more importantly, account for
        # additional fields in Run metadata JSON response.

    def start(self, batch_size: Optional[int] = None):
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

    def describe(self) -> Dict:
        """
        Retrieve the run metadata by querying the underlying DAO and return it as a dictionary.

        The underlying DAO method is expected to return
        a dictionary (JSON) representing the run's metadata response for flexibility (instead of a Run instance).
        """
        # TODO/TBD:  should we just return Run instance instead?

        result = self._run_dao.get_run(
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
        self._run_dao.delete_run(
            run_name=self.run_name,
            object_name=self.object_name,
            object_type=self.object_type,
        )
