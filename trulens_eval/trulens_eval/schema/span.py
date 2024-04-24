"""Span types.

These are roughly equivalent to RecordAppCall but abstract away specific method
information into type of call related to types of components.
"""

from __future__ import annotations

import contextvars
import datetime
from enum import Enum
import functools
import time
from typing import (ClassVar, Dict, Iterator, List, Mapping, Optional,
                    Sequence, Tuple, Union)

import opentelemetry
from opentelemetry.trace import status as trace_status
import opentelemetry.trace as ot_trace
import opentelemetry.trace.span as ot_span
from opentelemetry.util import types as ot_types
from opentelemetry.util._decorator import _agnosticcontextmanager
import pydantic

import trulens_eval

TTimestamp = int # uint64, nonoseconds since epoch, as per OpenTelemetry
TSpanID = int # as per OpenTelemetry
TTraceID = int # as per OpenTelemetry

# TODO: look into the open telemetry tracer/traceprovider api, ignoring for now.

class OTTracer(pydantic.BaseModel, ot_trace.Tracer):
    context: Optional[ot_trace.Context] = None

    instrumenting_module_name: str = "trulens_eval"
    instrumenting_library_version: str = trulens_eval.__version__

    def __init__(self):
        super().__init__()


    def start_span(
        self,
        name: str,
        context: Optional[ot_trace.Context] = None,
        kind: ot_trace.SpanKind = ot_trace.SpanKind.INTERNAL,
        attributes: ot_trace.types.Attributes = None,
        links: ot_trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> OTSpan:
    
        pass

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[ot_trace.Context] = None,
        kind: ot_trace.SpanKind = opentelemetry.trace.SpanKind.INTERNAL,
        attributes: ot_types.Attributes = None,
        links: ot_trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[OTSpan]:
        pass

class OTTracerProvider(ot_trace.TracerProvider):
    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
    ) -> OTTracer:

        return OTTracer(
            instrumenting_module_name=instrumenting_module_name,
            instrumenting_library_version=instrumenting_library_version or trulens_eval.__version__
        )


class OTSpan(pydantic.BaseModel, ot_span.Span):
    """Implementation of OpenTelemetry Span requirements.
    
    See also [OpenTelemetry Span](https://opentelemetry.io/docs/specs/otel/trace/api/#span).
    """

    _vendor: ClassVar[str] = "trulens_eval"
    @classmethod
    def _attr(self, name):
        return f"{self._vendor}@{name}"

    model_config = {
        'arbitrary_types_allowed': True,
        'use_attribute_docstrings': True
    }
    """Pydantic configuration."""

    name: str
    """Name of span."""

    status: trace_status.StatusCode = trace_status.StatusCode.UNSET
    """Status of the span as per OpenTelemetry Span requirements."""

    status_description: Optional[str] = None
    """Status description as per OpenTelemetry Span requirements."""

    start_timestamp: TTimestamp = pydantic.Field(default_factory=time.time_ns)
    """Timestamp when the span's activity started in nanoseconds since epoch."""

    end_timestamp: Optional[TTimestamp] = None
    """Timestamp when the span's activity ended in nanoseconds since epoch.

    None if not yet ended.
    """

    context: ot_span.SpanContext
    """Unique immutable identifier for the span."""

    events: List[Tuple[str, ot_types.Attributes, TTimestamp]] = pydantic.Field(default_factory=list)
    """Events recorded in the span."""

    links: Dict[ot_span.SpanContext, Mapping[str, ot_types.AttributeValue]] = pydantic.Field(default_factory=dict)
    """Relationships to other spans with attributes on each link."""

    attributes: Dict[str, ot_types.AttributeValue] = pydantic.Field(default_factory=dict)
    """Attributes of span."""

    def end(self, end_time: Optional[TTimestamp] = None):
        """See [end][opentelemetry.trace.span.Span.end]"""
        if end_time:
            self.end_timestamp = end_time
        else:
            self.end_timestamp = time.time_ns()

        self.status = trace_status.StatusCode.OK

    def get_span_context(self) -> ot_span.SpanContext:
        """See [end][opentelemetry.trace.span.Span.get_span_context]"""

        return self.context

    def set_attributes(self, attributes: Dict[str, ot_types.AttributeValue]) -> None:
        """See [end][opentelemetry.trace.span.Span.set_attributes]"""

        self.attributes.update(attributes)

    def set_attribute(self, key: str, value: ot_types.AttributeValue) -> None:
        """See [end][opentelemetry.trace.span.Span.set_attribute]"""

        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[int] = None
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.add_event]"""
        self.events.append((name, attributes, timestamp or time.time_ns()))

    def add_link(
        self,
        context: ot_span.SpanContext,
        attributes: ot_types.Attributes = None
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.add_link]"""

        if attributes is None:
            attributes = {}

        self.links[context] = attributes

    def update_name(self, name: str) -> None:
        """See [end][opentelemetry.trace.span.Span.update_name]."""

        self.name = name

    def is_recording(self) -> bool:
        """See [end][opentelemetry.trace.span.Span.is_recording]."""

        return self.status == trace_status.StatusCode.UNSET

    def set_status(
        self,
        status: Union[ot_span.Status, ot_span.StatusCode],
        description: Optional[str] = None
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.set_status]"""

        if isinstance(status, ot_span.Status):
            if description is not None:
                raise ValueError("Ambiguous status description provided both in `status.description` and in `description`.")
            
            self.status = status.status_code
            self.status_description = status.description
        else:
            self.status = status
            self.status_description = description

    def record_exception(
        self,
        exception: Exception,
        attributes: ot_types.Attributes = None,
        timestamp: Optional[TTimestamp] = None,
        escaped: bool = False
    ) -> None:
        """See [end][opentelemetry.trace.span.Span.record_exception]"""

        self.status = trace_status.StatusCode.ERROR

        self.add_event(
            self._attr("exception"),
            attributes,
            timestamp
        )

class DictNamespace(Dict[str, ot_types.AttributeValue]):
    """View into a dict with keys prefixed by some `namespace` string.
    
    Replicates the values without the prefix in self.
    """

    def __init__(self, parent: Dict, namespace: str, **kwargs):
        self.parent = parent
        self.namespace = namespace

    def __getitem__(self, key):
        return dict.__getitem__(self, key)
    
    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.parent[f"{self.namespace}.{key}"] = value

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        del self.parent[f"{self.namespace}.{key}"]

class SpanType(Enum):
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

class Span(OTSpan):
    """Base Span type.
    
    Smallest unit of recorded activity.
    """

    @property
    def span_id(self) -> TSpanID:
        """Identifier for the span."""

        return self.context.span_id

    @property
    def trace_id(self) -> TTraceID:
        """Identifier for the trace this span belongs to."""

        return self.context.trace_id

    @functools.cached_property
    def parent_span_id(self) -> Optional[TSpanID]:
        """Id of parent span if any.

        None if this is a root span.
        """

        for link in self.links:
            if link.trace_id == self.trace_id and link.attributes.get(self._attr("relationship")) == "parent":
                return link.span_id

        return None

    @property
    def tags(self) -> List[str]:
        """Tags associated with the span."""

        return self.attributes.get(self._attr("tags"), [])
    @tags.setter
    def tags(self, value: List[str]):
        self.attributes[self.attr("tags")] = value

    @property
    def span_type(self) -> SpanType:
        """Type of span."""

        return self.attributes.get(self._attr("span_type"), SpanType.OTHER)
    @span_type.setter
    def span_type(self, value: SpanType):
        self.attributes[self._attr("span_type")] = value

    attributes_metadata: DictNamespace[str, ot_types.AttributeValue] 
    # will be set as a DictNamespace indexing elements in attributes
    @property
    def metadata(self) -> DictNamespace[str, ot_types.AttributeValue]:
        return self.attributes_metadata

    @metadata.setter
    def metadata(self, value: Dict[str, str]):
        for k, v in value.items():
            self.attributes_metadata[k] = v

    # input: Dict[str, str] = pydantic.Field(default_factory=dict)
    # Make property

    # output: Dict[str, str] = pydantic.Field(default_factory=dict)
    # Make property

    def __init__(self, name: str, context: ot_span.SpanContext, **kwargs):
        attr_dict = {}
        kwargs['attributes'] = attr_dict
        kwargs['name'] = name
        kwargs['context'] = context

        kwargs['attributes_metadata'] = DictNamespace(parent={}, namespace="temp")
        # Temporary fake for validation in super.__init__ below.

        super().__init__(**kwargs)

        # Actual. This is needed as pydantic will copy attributes dict in init.
        self.attributes_metadata = DictNamespace(
            parent=self.attributes,
            namespace=self._attr("metadata")
        )

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
