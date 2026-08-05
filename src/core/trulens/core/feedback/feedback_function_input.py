from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class FeedbackFunctionInput:
    value: Optional[Any] = None
    span_id: Optional[Union[str, List[str]]] = None
    span_attribute: Optional[str] = None
    collect_list: bool = True
