from typing import Dict, List

from pydantic import BaseModel
from pydantic import Field


class RunConfig(BaseModel):
    name: str = Field(..., description="Unique run name.")
    description: str = Field(..., description="A description for the run.")
    run_label: str = Field(..., description="A label categorizing the run.")
    dataset_fqn: str = Field(
        ...,
        description="The fully qualified name of the dataset (e.g. 'db.schema.user_table_name_1').",
    )
    dataset_spec: Dict[str, str] = Field(
        ..., description="Mapping of dataset column names to dataset columns."
    )
    llm_judge_name: str = Field(
        ...,
        description="The name of the LLM judge to use (e.g. 'mistral-large2').",
    )


class Run:
    def __init__(self):
        pass

    def start(self, batch_size=1000):
        pass

    def status(self):
        pass

    def compute_metrics(self, metrics: List[str], params: dict):
        pass

    def cancel(self):
        pass
