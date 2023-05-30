from typing import Callable, Dict, Optional, Sequence, Union

import pydantic

from trulens_eval.util import JSON
from trulens_eval.util import JSONPath


class Chain(pydantic.BaseModel):
    chain_id: str
    chain: JSON  # langchain structure


class RecordChainCallFrame(pydantic.BaseModel):
    path: JSONPath
    method_name: str
    class_name: str
    module_name: str


class RecordCost(pydantic.BaseModel):
    n_tokens: Optional[int]
    cost: Optional[float]


class RecordChainCall(pydantic.BaseModel):
    """
    Info regarding each instrumented method call is put into this container.
    """

    # Call stack but only containing paths of instrumented chains/other objects.
    chain_stack: Sequence[RecordChainCallFrame]

    # Arguments to the instrumented method.
    args: Dict

    # Returns of the instrumented method.
    rets: Dict

    # Error message if call raised exception.
    error: Optional[str]

    # Timestamps tracking entrance and exit of the instrumented method.
    start_time: int
    end_int: int

    # Process id.
    pid: int

    # Thread id.
    tid: int


class Record(pydantic.BaseModel):
    record_id: str
    chain_id: str

    cost: RecordCost

    total_tokens: int
    total_cost: float

    calls: Sequence[
        RecordChainCall
    ]  # not the actual chain, but rather json structure that mirrors the chain structure


class FeedbackResult(pydantic.BaseModel):
    record_id: str
    chain_id: str
    feedback_id: Optional[str]

    results_json: JSON


Selection = Union[JSONPath, str]


class FeedbackDefinition(pydantic.BaseModel):
    # Implementation serialization info.
    imp_json: Optional[JSON] = pydantic.Field(exclude=True)

    # Id, if not given, unique determined from _json below.
    feedback_id: Optional[str] = None

    # Selectors, pointers into Records of where to get
    # arguments for `imp`.
    selectors: Optional[Dict[str, Selection]] = None

    # TODO: remove
    # JSON version of this object.
    feedback_json: Optional[JSON] = pydantic.Field(exclude=True)