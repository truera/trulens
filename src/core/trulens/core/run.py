from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from trulens.connectors.snowflake.dao.run import RunDao


class RunConfig(BaseModel):
    name: str = Field(..., description="Unique run name.")
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


class Run:
    def __init__(self) -> None:
        raise RuntimeError(
            "Run's initializer is not meant to be directly used."
        )

    name: str
    _run_dao: RunDao

    @classmethod
    def _ref(
        cls,
        run_dao: RunDao,
        name: str,
    ) -> "Run":
        self: "Run" = object.__new__(cls)
        self.name = name
        self._run_dao = run_dao
        return self

    def start(self, batch_size: Optional[int] = None):
        pass

    def status(self):
        pass

    def compute_metrics(self, metrics: List[str], params: dict):
        pass

    def cancel(self):
        pass

    def update(
        self, description: Optional[str] = None, label: Optional[str] = None
    ):
        """Only description and label are allowed to be updated at the moment.

        Args:
            description (Optional[str], optional): Run description.
            label (Optional[str], optional): Run label for grouping runs.
        """
        pass

    def describe(self):  # GET run from DPO
        pass
