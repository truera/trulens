"""Spans

These are roughly equivalent to `RecordAppCall` but abstract away specific method
information into type of call related to types of components.
"""

from __future__ import annotations

from enum import Enum
import functools
from logging import getLogger
from typing import Dict, List, Optional, Type

from opentelemetry.util import types as ot_types

from trulens_eval import trace as mod_trace
from trulens_eval.utils import containers as mod_container_utils

logger = getLogger(__name__)


class Span(mod_trace.OTSpan):
    """Base Span type.
    
    Smallest unit of recorded activity.
    """

    @property
    def span_id(self) -> mod_trace.TSpanID:
        """Identifier for the span."""

        return self.context.span_id

    @property
    def trace_id(self) -> mod_trace.TTraceID:
        """Identifier for the trace this span belongs to."""

        return self.context.trace_id

    @functools.cached_property
    def parent_context(self) -> Optional[mod_trace.HashableSpanContext]:
        """Context of parent span if any.

        None if this is a root span.
        """

        for link_context, link_attributes in self.links.items():
            if link_attributes.get(self.vendor_attr("relationship")) == "parent":
                return link_context

        return None

    @functools.cached_property
    def parent_span_id(self) -> Optional[mod_trace.TSpanID]:
        """Id of parent span if any.

        None if this is a root span.
        """

        parent_context = self.parent_context
        if parent_context is not None:
            return parent_context.span_id

        return None

    @property
    def tags(self) -> List[str]:
        """Tags associated with the span."""

        return self.attributes.get(self.vendor_attr("tags"), [])
    @tags.setter
    def tags(self, value: List[str]):
        self.attributes[self.attr("tags")] = value

    @property
    def span_type(self) -> SpanType:
        """Type of span."""

        return SpanType(self.attributes.get(self.vendor_attr("span_type"), SpanType.UNTYPED.name))

    @span_type.setter
    def span_type(self, value: SpanType):
        self.attributes[self.vendor_attr("span_type")] = value

    attributes_metadata: mod_container_utils.DictNamespace[str, ot_types.AttributeValue]
    # will be set as a DictNamespace indexing elements in attributes
    @property
    def metadata(self) -> mod_container_utils.DictNamespace[str, ot_types.AttributeValue]:
        return self.attributes_metadata

    @metadata.setter
    def metadata(self, value: Dict[str, str]):
        for k, v in value.items():
            self.attributes_metadata[k] = v

    # input: Dict[str, str] = pydantic.Field(default_factory=dict)
    # Make property
    # output: Dict[str, str] = pydantic.Field(default_factory=dict)
    # Make property

    def __init__(self, **kwargs):
        kwargs['attributes_metadata'] = mod_container_utils.DictNamespace(parent={}, namespace="temp")
        # Temporary fake for validation in super.__init__ below.

        super().__init__(**kwargs)

        # Actual. This is needed as pydantic will copy attributes dict in init.
        self.attributes_metadata = mod_container_utils.DictNamespace(
            parent=self.attributes,
            namespace=self.vendor_attr("metadata")
        )

        self.set_attribute(self.vendor_attr("span_type"), self.__class__.__name__)

    @staticmethod
    def attribute_property(name: str, typ: Type) -> property:
        """A property that will get/set values from `self.attributes`."""

        def getter(self):
            return self.attributes.get(self.vendor_attr(name))
        
        def setter(self, value):
            if not isinstance(value, typ):
                raise ValueError(f"Expected {typ} for {self.name} but got {type(value)}.")

            self.attributes[self.vendor_attr(name)] = value

        return property(getter, setter)

class SpanRecordAppCall(Span):
    """A Span which corresponds to single
    [RecordAppCall][trulens_eval.schema.record.RecordAppCall].
    """

class SpanUntyped(Span):
    """Generic span type.
    
    This represents spans that are being recorded but have not yet been
    determined to be of a particular type.
    """

    pass

class SpanRoot(Span):
    """A root span encompassing some collection of spans.

    Does not indicate any particular activity by itself beyond its children.
    """
    pass

class SpanRetriever(Span):
    """A retrieval."""

    input_text = Span.attribute_property("input_text", str)
    """Input text whose related contexts are being retrieved."""

    input_embedding = Span.attribute_property("input_embedding", list)#List[float])
    """Embedding of the input text."""

    distance_type = Span.attribute_property("distance_type", str)
    """Distance function used for ranking contexts."""

    num_contexts = Span.attribute_property("num_contexts", int)
    """The number of contexts requested, not necessarily retrieved."""

    retrieved_contexts = Span.attribute_property("retrieved_contexts", list)#List[str])
    """The retrieved contexts."""

class SpanReranker(Span):
    pass

class SpanLLM(Span):
    model_name = Span.attribute_property("model_name", str)
    """The model name of the LLM."""

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

class SpanType(Enum):
    """Span types.
    
    This is a bit redundant with the span type hierarchy above. It is here for
    convenience of looking up types in means other than `__class__` or via
    `isinstance`.
    """

    def to_class(self) -> Type[Span]:
        """Convert to the class for this type."""

        if hasattr(mod_trace.span, self.value):
            return getattr(mod_trace.span, self.value)
        
        raise ValueError(f"Span type {self.value} not found in module.")

    UNTYPED = SpanUntyped.__name__
    """See [SpanUntyped][trulens_eval.trace.span.SpanUntyped]."""

    ROOT = SpanRoot.__name__
    """See [SpanRoot][trulens_eval.trace.span.SpanRoot]."""

    RETRIEVER = SpanRetriever.__name__
    """See [SpanRetriever][trulens_eval.trace.span.SpanRetriever]."""

    RERANKER = SpanReranker.__name__
    """See [SpanReranker][trulens_eval.trace.span.SpanReranker]."""

    LLM = SpanLLM.__name__
    """See [SpanLLM][trulens_eval.trace.span.SpanLLM]."""

    EMBEDDING = SpanEmbedding.__name__
    """See [SpanEmbedding][trulens_eval.trace.span.SpanEmbedding]."""

    TOOL = SpanTool.__name__
    """See [SpanTool][trulens_eval.trace.span.SpanTool]."""

    AGENT = SpanAgent.__name__
    """See [SpanAgent][trulens_eval.trace.span.SpanAgent]."""

    TASK = SpanTask.__name__
    """See [SpanTask][trulens_eval.trace.span.SpanTask]."""

    OTHER = SpanOther.__name__
    """See [SpanOther][trulens_eval.trace.span.SpanOther]."""
