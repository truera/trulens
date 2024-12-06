# ruff: noqa: E402

"""Tracer for OTEL tracing.

Adds TruLens specific features on top of the minimal OTEL Tracer.

!!! Note
    Most of the module is EXPERIMENTAL(otel_tracing) though it includes some existing
    non-experimental classes moved here to resolve some circular import issues.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import sys
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Optional,
    Set,
    Type,
    TypeVar,
)

from opentelemetry.util import types as types_api
import pydantic
from trulens.core.schema import types as types_schema
from trulens.core.utils import python as python_utils
from trulens.experimental.otel_tracing.core.trace import otel as core_otel
from trulens.experimental.otel_tracing.core.trace import span as core_span

if TYPE_CHECKING:
    # Need to model_rebuild classes thast use any of these:
    from trulens.experimental.otel_tracing.core.trace import (
        context as core_context,
    )

if sys.version_info < (3, 9):
    from functools import lru_cache as fn_cache
else:
    from functools import cache as fn_cache

T = TypeVar("T")
R = TypeVar("R")  # callable return type
E = TypeVar("E")  # iterator/generator element type
S = TypeVar("S")  # span type

logger = logging.getLogger(__name__)


class Tracer(core_otel.Tracer):
    """TruLens additions on top of [OTEL Tracer][opentelemetry.trace.Tracer]."""

    # TODO: Create a Tracer that does not record anything. Can either be a
    # setting to this tracer or a separate "NullTracer". We need non-recording
    # users to not incur much overhead hence need to be able to disable most of
    # the tracing logic when appropriate.

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    # Overrides core_otel.Tracer._span_class
    _span_class: Type[core_otel.Span] = pydantic.PrivateAttr(
        default_factory=lambda: core_span.Span
    )

    @property
    def spans(self) -> Dict[core_context.SpanContext, core_otel.Span]:
        return self._tracer_provider.spans

    @property
    def current_span(self) -> Optional[core_otel.Span]:
        if (context := self.current_span_context) is None:
            return None

        return self.spans.get(context)

    def start_span(self, *args, **kwargs):
        """Like OTEL start_span except also keeps track of the span just created."""

        new_span = super().start_span(*args, **kwargs)

        self.spans[new_span.context] = new_span

        return new_span

    @staticmethod
    def find_each_child(
        span: core_span.Span, span_filter: Callable
    ) -> Iterable[core_span.Span]:
        """For each family rooted at each child of this span, find the top-most
        span that satisfies the filter."""

        for child_span in span.children_spans:
            if span_filter(child_span):
                yield child_span
            else:
                yield from Tracer.find_each_child(child_span, span_filter)


class TracerProvider(
    core_otel.TracerProvider, metaclass=python_utils.PydanticSingletonMeta
):
    """TruLens additions on top of [OTEL TracerProvider][opentelemetry.trace.TracerProvider]."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    _trace_id: types_schema.TraceID.PY_TYPE = pydantic.PrivateAttr(
        default_factory=types_schema.TraceID.default_py
    )

    def __str__(self):
        # Pydantic will not print anything useful otherwise.
        return f"{self.__module__}.{type(self).__name__}()"

    @property
    def trace_id(self) -> types_schema.TraceID.PY_TYPE:
        return self._trace_id

    # Overrides core_otel.TracerProvider._tracer_class
    _tracer_class: Type[Tracer] = pydantic.PrivateAttr(default=Tracer)

    _tracers: Dict[str, Tracer] = pydantic.PrivateAttr(default_factory=dict)

    _spans: Dict[core_context.SpanContext, core_otel.Span] = (
        pydantic.PrivateAttr(default_factory=dict)
    )

    @property
    def spans(self) -> Dict[core_context.SpanContext, core_otel.Span]:
        return self._spans

    _exported_map: Dict[Hashable, Set[core_context.SpanContext]] = (
        pydantic.PrivateAttr(default_factory=lambda: defaultdict(set))
    )
    """NON-STANDARD: Each sink (hashable) is mapped to the set of span contexts
    it has received.

    This is to prevent saving the same span twice or exporting it twice. Due to
    the recording context nature of TruLens, the same spans can be processed for
    multiple apps/contexts but we don't want to write them more than once.
    """

    def was_exported_to(
        self,
        context: core_context.SpanContext,
        to: Hashable,
        mark_exported: bool = False,
    ) -> bool:
        """Determine whether the given span context has been exported to the
        given sink.

        Optionally marks the span context as exported.
        """

        ret = context in self._exported_map[to]

        if mark_exported:
            self._exported_map[to].add(context)

        return ret

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        attributes: Optional[types_api.Attributes] = None,
    ):
        if instrumenting_module_name in self._tracers:
            return self._tracers[instrumenting_module_name]

        tracer = super().get_tracer(
            instrumenting_module_name=instrumenting_module_name,
            instrumenting_library_version=instrumenting_library_version,
            attributes=attributes,
            schema_url=schema_url,
        )

        self._tracers[instrumenting_module_name] = tracer

        return tracer


@fn_cache
def trulens_tracer_provider():
    """Global tracer provider.
    All trulens tracers are made by this provider even if a different one is
    configured for OTEL.
    """

    return TracerProvider()


def was_exported_to(
    context: core_context.SpanContext, to: Hashable, mark_exported: bool = False
):
    """Determine whether the given span context has been exported to the given sink.

    Optionally marks the span context as exported.
    """

    return trulens_tracer_provider().was_exported_to(context, to, mark_exported)


@fn_cache
def trulens_tracer():
    from trulens.core import __version__

    return trulens_tracer_provider().get_tracer(
        instrumenting_module_name="trulens.experimental.otel_tracing.core.trace",
        instrumenting_library_version=__version__,
    )
