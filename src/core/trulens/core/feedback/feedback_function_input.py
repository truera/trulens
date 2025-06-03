from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FeedbackFunctionInput:
    value: Optional[Any] = None
    span_id: Optional[str] = None
    span_attribute: Optional[str] = None
    collect_list: bool = True
