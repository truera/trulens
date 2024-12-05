# ruff: noqa: E402

"""Spans extending OTEL functionality for TruLens."""

from __future__ import annotations

import inspect
import os
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import uuid

from opentelemetry.semconv.resource import ResourceAttributes
import pydantic
from trulens.core._utils.pycompat import ReferenceType
from trulens.core._utils.pycompat import TypeAlias
from trulens.core._utils.pycompat import WeakSet
from trulens.core.schema import base as base_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.experimental.otel_tracing.core.trace import context as core_context
from trulens.experimental.otel_tracing.core.trace import otel as core_otel
from trulens.experimental.otel_tracing.core.trace import trace as core_trace
from trulens.otel.semconv import trace as truconv

if TYPE_CHECKING:
    from trulens.core import app as core_app

T = TypeVar("T")
R = TypeVar("R")  # callable return type
E = TypeVar("E")  # iterator/generator element type


class AttributeProperty(property, Generic[T]):
    """Property that stores a serialized version its value in the attributes
    dictionary.

    Validates default and on assignment. This is meant to be used only in
    trulens Span instances (or subclasses).

        Args:
            name: The name of the property. The key used for storage will be
                this with the vendor prefix.

            typ: The type of the property.

            typ_factory: A factory function that returns the type of the
                property. This can be used for forward referenced types.

            default: The default value of the property.

            default_factory: A factory function that returns the default value
                of the property. This can be used for defaults that make use of
                forward referenced types.
    """

    def __init__(
        self,
        name: str,
        typ: Optional[Type[T]] = None,
        typ_factory: Optional[Callable[[], Type[T]]] = None,
        default: Optional[T] = None,
        default_factory: Optional[Callable[[], T]] = None,
    ):
        self.name = name
        self.typ = typ
        self.typ_factory = typ_factory
        self.default = default
        self.default_factory = default_factory

        self.field_name: Optional[str] = None

        self.forward_initialized = False

    def init_forward(self):
        if self.forward_initialized:
            return

        self.forward_initialized = True

        if self.typ is None and self.typ_factory is not None:
            self.typ = self.typ_factory()

        if self.default is None and self.default_factory is not None:
            self.default = self.default_factory()

        if self.typ is None and self.default is not None:
            self.typ = type(self.default)

        if self.typ is None:
            self.tadapter = None
        else:
            try:
                self.tadapter = pydantic.TypeAdapter(self.typ)

                if self.default is not None:
                    self.tadapter.validate_python(self.default)

            except pydantic.PydanticSchemaGenerationError:
                self.tadapter = None

    def fget(self, obj: Any) -> Optional[T]:
        return self.__get__(obj, obj.__class__)

    def __get__(self, obj: Any, objtype: Optional[Type[T]]) -> Optional[T]:  # type: ignore # noqa: F821
        if obj is None:
            return self

        self.init_forward()
        return obj._attributes.get(self.name, self.default)

    def __set__(self, obj, value: T) -> None:
        self.init_forward()

        if self.tadapter is not None:
            try:
                self.tadapter.validate_python(value)
            except pydantic.ValidationError as e:
                raise ValueError(
                    f"Invalid value for attribute {self.field_name}: {e}"
                )

        obj._attributes[self.name] = value
        obj.attributes[self.name] = json_utils.jsonify(value)

    def __delete__(self, obj):
        del obj._attributes[self.name]
        del obj.attributes[self.name]

    def __set_name__(self, cls, name):
        self.field_name = name

        if name in cls.__annotations__:
            # If type is specified in annotation, take it from there.
            self.typ = cls.__annotations__[name]
            self.tadapter = pydantic.TypeAdapter(self.typ)

            # Update the recorded return type as well.
            # TODO: cannot do this at this point as the below dict is not yet populated
            # if name in cls.model_computed_fields:
            #     cls.model_computed_fields[name].return_type = self.typ

            # Have to remove it as pydantic will complain about overriding fields with computed fields.
            del cls.__annotations__[name]


class WithAttributeProperties(pydantic.BaseModel):
    _attributes: Dict[str, Any] = pydantic.PrivateAttr(default_factory=dict)
    """Non-serialized values of named fields defined by `attribute_property`.

    These are mirrored with serialized versions in `attributes`.
    """

    @staticmethod
    def attribute_property_factory(base: str) -> Callable:
        def prop_factory(
            name: str,
            typ: Optional[Type[T]] = None,
            typ_factory: Optional[Callable[[], Type[T]]] = None,
            default: Optional[T] = None,
            default_factory: Optional[Callable[[], T]] = None,
        ) -> property:
            return Span.attribute_property(
                name=base + "." + name,
                typ=typ,
                typ_factory=typ_factory,
                default=default,
                default_factory=default_factory,
            )

        return prop_factory

    @staticmethod
    def attribute_property(
        name: str,
        typ: Optional[Type[T]] = None,
        typ_factory: Optional[Callable[[], Type[T]]] = None,
        default: Optional[T] = None,
        default_factory: Optional[Callable[[], T]] = None,
    ) -> property:
        """See AttributeProperty."""

        return pydantic.computed_field(
            AttributeProperty(name, typ, typ_factory, default, default_factory),
            return_type=typ,
        )


class Span(core_otel.Span, WithAttributeProperties):
    """TruLens additions on top of OTEL spans.

    Note that in this representation, we keep track of the tracer that produced
    the instance and have properties to access other spans from that tracer,
    like the parent. This make traversing lives produced in this process a bit
    easier.
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,  # model_validate will fail without this
    )

    def __str__(self):
        return (
            f"{type(self).__name__}({self.name}, {self.context}->{self.parent})"
        )

    def __repr__(self):
        return str(self)

    @property
    def parent_span(self) -> Optional[Span]:
        if self.parent is None:
            return None

        if self._tracer is None:
            return None

        if (span := self._tracer.spans.get(self.parent)) is None:
            return None

        return span

    _children_spans: List[Span] = pydantic.PrivateAttr(default_factory=list)

    @property
    def children_spans(self) -> List[Span]:
        return self._children_spans

    error: Optional[Exception] = pydantic.Field(None)
    """Optional error if the observed computation raised an exception."""

    record_ids = WithAttributeProperties.attribute_property(
        truconv.SpanAttributes.RECORD.RECORD_IDS,
        typ=Dict[types_schema.AppID, types_schema.RecordID],
        default_factory=dict,
    )
    """App id to record id map.

    This is because the same span might represent part of the trace of different
    records because more than one app is tracing.

    This will not be filled in if the span was produced outside of a recording
    context.
    """

    app_ids = WithAttributeProperties.attribute_property(
        truconv.SpanAttributes.RECORD.APP_IDS,
        typ=Set[types_schema.AppID],
        default_factory=set,
    )
    """Apps recording this span."""

    def __init__(self, **kwargs):
        # Convert any contexts to our hashable context class:
        if (context := kwargs.get("context")) is not None:
            kwargs["context"] = core_context.SpanContext.of_contextlike(context)
        if (parent := kwargs.get("parent", None)) is not None:
            kwargs["parent"] = core_context.SpanContext.of_contextlike(parent)

        super().__init__(**kwargs)

        if (parent_span := self.parent_span) is not None:
            if isinstance(parent_span, Span):
                parent_span.children_spans.append(self)

        self._init_attribute_properties(kwargs)

    def _init_attribute_properties(self, kwargs):
        # Attribute_property fields are not automatically set from kwargs or
        # from attributes. We set them here.

        for name in kwargs.keys():
            if not hasattr(self, name):
                raise RuntimeWarning("Unknown field in kwargs: " + name)

        all_computed_fields = dict(self.model_computed_fields.items())

        for name, field in all_computed_fields.items():
            if not isinstance(field.wrapped_property, AttributeProperty):
                # Only cover our AttributeProperty.
                continue

            prop = field.wrapped_property
            attribute_name = prop.name

            if (val := kwargs.get(name)) is not None:
                # Got from kwargs.
                pass
            else:
                if (
                    val := self.attributes.get(attribute_name, None)
                ) is not None:
                    # Got from OTEL attributes.
                    pass
                else:
                    # Get from defaults specified on AttributeProperty.
                    val = prop.default
                    if prop.default_factory is not None:
                        val = prop.default_factory()

            setattr(self, name, val)

    def iter_ancestors(self) -> Iterable[Span]:
        """Iterate over all ancestors of this span."""

        yield self

        if self.parent_span is not None:
            yield from self.parent_span.iter_ancestors()

    def has_ancestor_of_type(self, span_type: Type[Span]) -> bool:
        """Check if this span has an ancestor of the given type."""

        for ancestor in self.iter_ancestors():
            if isinstance(ancestor, span_type):
                return True

        return False

    def iter_children(
        self,
        transitive: bool = True,
        matching: Optional[SpanFilterLike] = None,
    ) -> Iterable[Span]:
        """Iterate over all spans that are children of this span.

        Args:
            transitive: Iterate recursively over children.

            matching: Optional filter function to apply to each child span.
        """

        matching = _filter_of_spanfilterlike(matching)

        for child_span in self.children_spans:
            if matching(child_span):
                yield child_span
            if transitive:
                yield from child_span.iter_children(
                    transitive=transitive,
                    matching=matching,
                )

    def first_child(
        self,
        transitive: bool = True,
        matching: Optional[SpanFilterLike] = None,
    ) -> Optional[Span]:
        """Get the first child span that passes the filter."""

        matching = _filter_of_spanfilterlike(matching)

        try:
            return next(iter(self.iter_children(transitive, matching)))
        except StopIteration:
            return None

    def iter_family(
        self, matching: Optional[SpanFilterLike] = None
    ) -> Iterable[Span]:
        """Iterate itself and all children transitively."""

        matching = _filter_of_spanfilterlike(matching)

        yield from self.iter_children(transitive=True, matching=matching)

    def cost_tally(self) -> base_schema.Cost:
        """Total costs of this span and all its transitive children."""

        total = base_schema.Cost()

        for span in self.iter_family():
            if isinstance(span, WithCost) and span.cost is not None:
                total += span.cost

        return total


SpanFilterLike: TypeAlias = Union[Type[Span], Callable[[Span], bool]]
"""Filter for spans.

Either a span type (interpreted as an `isinstance check`) or a callable from
span to bool. Produces a callable from span to bool.
"""


def _filter_of_spanfilterlike(
    filter: Optional[SpanFilterLike],
) -> Callable[[Span], bool]:
    """Create a filter function from a SpanFilterLike.

    Defaults to filter that accepts all spans.
    """

    if filter is None:
        return lambda s: True

    if isinstance(filter, type):
        return lambda s: isinstance(s, filter)

    return filter


class LiveSpan(Span):
    """A a span type that indicates that it contains live python objects.

    It is to be converted to a non-live span before being output to the user or
    otherwise.
    """

    live_apps: WeakSet[core_app.App] = pydantic.Field(
        default_factory=WeakSet, exclude=True
    )  # Any = App
    """Apps for which this span is recording trace info for.

    WeakSet to prevent memory leaks.

    Note that this will not be filled in if this span was produced outside of an
    app recording context.
    """


class RecordingContextSpan(LiveSpan):
    """Tracks the context of an app used as a context manager."""

    live_recording: Optional[Any] = pydantic.Field(None, exclude=True)
    # TODO: app.RecordingContext # circular import issues

    live_app: Optional[ReferenceType[core_app.App]] = pydantic.Field(
        None, exclude=True
    )

    def otel_resource_attributes(self) -> Dict[str, Any]:
        ret = super().otel_resource_attributes()

        ret[ResourceAttributes.SERVICE_NAME] = (
            self.live_recording.app.app_name
            if self.live_recording is not None
            else None
        )

        return ret

    # override to also call _finalize_recording .
    def end(self, *args, **kwargs):
        super().end(*args, **kwargs)

        self._finalize_recording()

    def _finalize_recording(self):
        assert self.live_recording is not None

        app = self.live_recording.app

        for span in core_trace.Tracer.find_each_child(
            span=self, span_filter=lambda s: isinstance(s, LiveRecordRoot)
        ):
            app._on_new_root_span(recording=self.live_recording, root_span=span)

        app._on_new_recording_span(recording_span=self)


class LiveSpanCall(LiveSpan):
    """Track a function call."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    call_id = WithAttributeProperties.attribute_property(
        truconv.SpanAttributes.CALL.CALL_ID,
        Optional[uuid.UUID],
        default_factory=uuid.uuid4,
    )
    """Unique call identifiers."""

    process_id = WithAttributeProperties.attribute_property(
        truconv.SpanAttributes.CALL.PROCESS_ID, int, default_factory=os.getpid
    )
    """Process ID of the call."""

    thread_id = WithAttributeProperties.attribute_property(
        truconv.SpanAttributes.CALL.THREAD_ID,
        int,
        default_factory=threading.get_native_id,
    )
    """Thread ID of the call."""

    call_error = WithAttributeProperties.attribute_property(
        truconv.SpanAttributes.CALL.ERROR, Optional[Exception], default=None
    )
    """Optional error if the called function raised an exception."""

    live_sig: Optional[inspect.Signature] = pydantic.Field(None, exclude=True)
    """Called function's signature."""

    live_obj: Optional[Any] = pydantic.Field(None, exclude=True)
    """Self object if method call."""

    live_cls: Optional[Type] = pydantic.Field(None, exclude=True)
    """Class if method/static/class method call."""

    live_func: Optional[Callable] = pydantic.Field(None, exclude=True)
    """Function object."""

    live_args: Optional[Tuple[Any, ...]] = pydantic.Field(None, exclude=True)
    """Positional arguments to the function call."""

    live_kwargs: Optional[Dict[str, Any]] = pydantic.Field(None, exclude=True)
    """Keyword arguments to the function call."""

    live_bound_arguments: Optional[inspect.BoundArguments] = pydantic.Field(
        None, exclude=True
    )
    """Bound arguments to the function call if can be bound."""

    live_ret: Optional[Any] = pydantic.Field(None, exclude=True)
    """Return value of the function call.

    Exclusive with `error`.
    """

    live_error: Optional[Any] = pydantic.Field(None, exclude=True)
    """Error raised by the function call.

    Exclusive with `ret`.
    """


class LiveRecordRoot(LiveSpan):
    """Wrapper for first app calls, or "records".

    Children spans of type `LiveSpan` are expected to contain the app named here
    in their `live_apps` field.
    """

    live_app: Optional[ReferenceType[core_app.App]] = pydantic.Field(
        None, exclude=True
    )
    """The app for which this is the root call.

    Value must be included in children's `live_apps` field.
    """

    record_id = WithAttributeProperties.attribute_property(
        truconv.SpanAttributes.RECORD_ROOT.RECORD_ID,
        types_schema.TraceRecordID.PY_TYPE,
    )
    """Unique identifier for this root call or what is called a "record".

    Note that this is different from `record_ids` though this
    `record_id` will be included in `record_ids` and will be included in
    children's `record_ids` fields.

    Note that a record root cannot be a distributed call hence there is no
    non-live record root.
    """


class WithCost(LiveSpan):
    """Mixin to indicate the span has costs tracked."""

    cost = WithAttributeProperties.attribute_property(
        truconv.SpanAttributes.COST.COST,
        base_schema.Cost,
        default_factory=base_schema.Cost,
    )
    """Cost of the computation spanned."""

    live_endpoint: Optional[Any] = pydantic.Field(
        None, exclude=True
    )  # Any actually core_endpoint.Endpoint
    """Endpoint handling cost extraction for this span/call."""


class LiveSpanCallWithCost(LiveSpanCall, WithCost):
    pass
