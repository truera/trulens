# ruff: noqa: E402

""" """

from __future__ import annotations

import inspect
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from trulens.core.schema import base as base_schema
from trulens.core.schema import record as record_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.experimental.otel_tracing.core.trace import otel as core_otel
from trulens.experimental.otel_tracing.core.trace import sem as core_sem
from trulens.experimental.otel_tracing.core.trace import span as core_span

logger = logging.getLogger(__name__)


def _fill_stacks(
    span: core_span.Span,
    get_method_path: Callable,
    span_stacks: Dict[core_otel.Span, List[record_schema.RecordAppCallMethod]],
    stack: Optional[List[record_schema.RecordAppCallMethod]] = None,
):
    """Populate span_stacks with a mapping of span to call stack for
    backwards compatibility with records.

    Args:
        span: Span to start from.

        get_method_path: Function that looks up lens of a given
            obj/function. This is an WithAppCallbacks method.

        span_stacks: Mapping of span to call stack. This will be modified by
            this method.

        stack: Current call stack. Recursive calls will build this up.
    """
    if stack is None:
        stack = []

    if isinstance(span, core_span.LiveSpanCall):
        if span.live_func is None:
            print(span.attributes)
            raise ValueError(f"Span {span} has no function.")

        path = get_method_path(obj=span.live_obj, func=span.live_func)

        if path is None:
            logger.warning(
                "No path found for %s in %s.", span.live_func, span.live_obj
            )
            path = serial_utils.Lens().static

        if inspect.ismethod(span.live_func):
            # This is a method.
            frame_ident = record_schema.RecordAppCallMethod(
                path=path,
                method=pyschema_utils.Method.of_method(
                    span.live_func, obj=span.live_obj, cls=span.live_cls
                ),
            )
        elif inspect.isfunction(span.live_func):
            # This is a function, not a method.
            frame_ident = record_schema.RecordAppCallMethod(
                path=path,
                method=None,
                function=pyschema_utils.Function.of_function(span.live_func),
            )
        else:
            raise ValueError(f"Unexpected function type: {span.live_func}")

        stack = stack + [frame_ident]
        span_stacks[span] = stack

    for subspan in span.iter_children(transitive=False):
        _fill_stacks(
            subspan,
            stack=stack,
            get_method_path=get_method_path,
            span_stacks=span_stacks,
        )


def _call_of_spancall(
    span: core_span.LiveSpanCall,
    stack: List[record_schema.RecordAppCallMethod],
) -> record_schema.RecordAppCall:
    """Convert a LiveSpanCall to a RecordAppCall."""

    args = (
        dict(span.live_bound_arguments.arguments)
        if span.live_bound_arguments is not None
        else {}
    )
    if "self" in args:
        del args["self"]  # remove self

    assert span.start_timestamp is not None
    if span.end_timestamp is None:
        logger.warning(
            "Span %s has no end timestamp. It might not have yet finished recording.",
            span,
        )

    return record_schema.RecordAppCall(
        call_id=str(span.call_id),
        stack=stack,
        args={k: json_utils.jsonify(v) for k, v in args.items()},
        rets=json_utils.jsonify(span.live_ret),
        error=str(span.live_error),
        perf=base_schema.Perf(
            start_time=span.start_timestamp,
            end_time=span.end_timestamp,
        ),
        pid=span.process_id,
        tid=span.thread_id,
    )


def record_of_root_span(
    recording: Any, root_span: core_span.LiveRecordRoot
) -> record_schema.Record:
    """Convert a root span to a record.

    This span has to be a call span so we can extract things like main input and output.
    """

    assert isinstance(root_span, core_span.LiveRecordRoot), type(root_span)

    app = recording.app

    # Use the record_id created during tracing.
    record_id = root_span.record_id

    span_stacks: Dict[
        core_otel.Span, List[record_schema.RecordAppCallMethod]
    ] = {}

    _fill_stacks(
        root_span,
        span_stacks=span_stacks,
        get_method_path=app.get_method_path,
    )

    if root_span.end_timestamp is None:
        raise RuntimeError(f"Root span has not finished recording: {root_span}")

    root_perf = base_schema.Perf(
        start_time=root_span.start_timestamp,
        end_time=root_span.end_timestamp,
    )

    total_cost = root_span.cost_tally()

    calls = []
    spans = [core_sem.TypedSpan.semanticize(root_span)]

    root_call_span = None
    for span in root_span.iter_children():
        if isinstance(span, core_span.LiveSpanCall):
            calls.append(_call_of_spancall(span, stack=span_stacks[span]))
            root_call_span = root_call_span or span

        spans.append(core_sem.TypedSpan.semanticize(span))

    if root_call_span is None:
        raise ValueError("No call span found under trace root span.")

    bound_arguments = root_call_span.live_bound_arguments
    main_error = root_call_span.live_error

    if bound_arguments is not None:
        main_input = app.main_input(
            func=root_call_span.live_func,
            sig=root_call_span.live_sig,
            bindings=root_call_span.live_bound_arguments,
        )
        if main_error is None:
            main_output = app.main_output(
                func=root_call_span.live_func,
                sig=root_call_span.live_sig,
                bindings=root_call_span.live_bound_arguments,
                ret=root_call_span.live_ret,
            )
        else:
            main_output = None
    else:
        main_input = None
        main_output = None

    record = record_schema.Record(
        record_id=record_id,
        app_id=app.app_id,
        main_input=json_utils.jsonify(main_input),
        main_output=json_utils.jsonify(main_output),
        main_error=json_utils.jsonify(main_error),
        calls=calls,
        perf=root_perf,
        cost=total_cost,
        experimental_otel_spans=spans,
    )

    return record
