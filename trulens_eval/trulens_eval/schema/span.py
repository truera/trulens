"""Span types.

These are roughly equivalent to RecordAppCall but abstract away specific method
information into type of call related to types of components.
"""

import datetime
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Union

import opentelemetry.trace.span as ot_span
from opentelemetry.util import types as ot_types
import pydantic

SpanID = str
TraceID = str
# str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]

class DictNamespace(Dict[str, ot_types.AttributeValue]):

    def __init__(self, parent: Dict, namespace: str):
        super().__init__(**kwargs)

        self.parent = parent
        self.namespace = namespace

    def __getitem__(self, key):
        return self.parent[f"{self.namespace}.{key}"]
    
    def __setitem__(self, key, value):
        self.parent[f"{self.namespace}.{key}"] = value

    def __delitem__(self, key):
        del self.parent[f"{self.namespace}.{key}"]


class SpanStatus(Enum, str):
    """Span status."""

    INCOMPLETE = "incomplete"
    """Span's activity has not yet finished."""

    SUCCESS = "success"
    """Span finished without error."""

    ERROR = "error"
    """Span finished with an error."""

class Span(pydantic.BaseModel, ot_span.Span):
    """Base Span type.
    
    Smallest unit of recorded activity.
    """

    name: str
    """Name of the span."""

    span_id: SpanID
    """Unique identifier for the span."""

    trace_id: TraceID
    """Unique identifier for the trace this span belongs to."""

    parent_span_id: Optional[SpanID] = None
    """Id of parent span if any.

    None if this is a root span.
    """

    start_timestamp: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    """Datetime when the span's activity started."""

    end_timestamp: Optional[datetime.datetime] = None
    """Datetime when the span's activity ended.
    
    None if not yet ended.
    """

    status: SpanStatus = SpanStatus.INCOMPLETE
    """Status of the span's activity."""

    tags: List[str] = pydantic.Field(default_factory=list)

    # span_type: SpanType = SpanType.OTHER

    _attributes_metadata: Dict[str, ot_types.AttributeValue] 
    # will be set as a DictNamespace indexing elements in attributes

    @property
    def metadata(self) -> Dict[str, ot_types.AttributeValue]:
        return self._attributes_metadata

    @metadata.setter
    def metadata(self, value: Dict[str, str]):
        for k, v in value.items():
            self._attributes_metadata[k] = v

    def __init__(self, **kwargs):

        kwargs['_attributes_metadata'] = DictNamespace(
            parent=self.attributes,
            namespace="trulens_eval@metadata"
        )

        super().__init__(**kwargs)

    input: Dict[str, str] = pydantic.Field(default_factory=dict)

    output: Dict[str, str] = pydantic.Field(default_factory=dict)

    # for implementing ot_span.Span requirements:
    
    attributes: Dict[str, ot_types.AttributeValue] = pydantic.Field(default_factory=dict)

    context: Optional[ot_span.SpanContext] = None

    # begin ot_span.Span requirements

    
    def end(self, end_time: Optional[int] = None):
        """See [end][opentelemetry.trace.span.Span.end]"""
        if end_time:
            self.end_timestamp = datetime.datetime.fromtimestamp(end_time)
        else:
            self.end_timestamp = datetime.datetime.now()

    def get_span_context(self) -> ot_span.SpanContext:
        """See [end][opentelemetry.trace.span.Span.get_span_context]"""

        return self.context

    def set_attributes(self, attributes: Dict[str, ot_types.AttributeValue]) -> None:
        """See [end][opentelemetry.trace.span.Span.set_attributes]"""
        raise NotImplementedError("set_attributes not implemented")

    def set_attribute(self, key: str, value: ot_types.AttributeValue) -> None:
        """See [end][opentelemetry.trace.span.Span.set_attribute]"""
        raise NotImplementedError("set_attribute not implemented")

    def add_event(
        self,
        name: str,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[int] = None
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.add_event]"""
        raise NotImplementedError("add_event not implemented")

    def add_link(
        self,
        context: ot_span.SpanContext,
        attributes: ot_types.Attributes = None
    ) -> None:
        raise NotImplementedError("add_link not implemented")

    def update_name(self, name: str) -> None:
        raise NotImplementedError("update_name not implemented")

    def is_recording(self) -> bool:
        raise NotImplementedError("is_recording not implemented")

    def set_status(
        self,
        status: Union[ot_span.Status, ot_span.StatusCode],
        description: Optional[str] = None
    ) -> None:
        raise NotImplementedError("set_status not implemented")

    def record_exception(
        self,
        exception: Exception,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[int] = None,
        escaped: bool = False
    ) -> None:
        raise NotImplementedError("record_exception not implemented")
    
    # end ot_span.Span requirements

class SpanRoot(Span):
    pass

class SpanRetriever(Span):
    pass

class SpanReranker(Span):
    pass

class SpanLLM(Span):
    pass

class SpanEmbedding(Span):
    pass

class SpanTool(Span):
    pass

class SpanAgent(Span):
    pass

class SpanTask(Span):
    pass

class SpanOther(Span):
    pass


class SpanType(Enum, str):
    """Span types."""

    ROOT = "root"

    RETRIEVER = "retriever"

    RERANKER = "reranker"

    LLM = "llm"

    EMBEDDING = "embedding"

    TOOL = "tool"

    AGENT = "agent"

    TASK = "task"

    OTHER = "other"