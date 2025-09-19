"""LlamaIndex Workflow instrumentation.

This implements automatic step tracing by instrumenting the workflow executor
and the step callable classes at class level, similar to how TruGraph
instruments LangGraph's TaskFunction.__call__.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
from typing import Any, Callable, ClassVar, Optional

from pydantic import Field
from trulens.core import app as core_app
from trulens.core.otel.instrument import instrument_method
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils
from trulens.otel.semconv.trace import SpanAttributes  # type: ignore

logger = logging.getLogger(__name__)


def _safe_str(obj: Any) -> Optional[str]:
    try:
        return None if obj is None else str(obj)
    except Exception:
        return None


def _to_json_safe(val: Any) -> Any:
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [_to_json_safe(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _to_json_safe(v) for k, v in val.items()}
    return _safe_str(val)


def _serialize_event(ev: Any) -> Any:
    """Best-effort structured serialization of LlamaIndex Event objects."""
    if ev is None:
        return None
    # If StopEvent-like, prefer its result if accessible without calling.
    res = _extract_stop_result(ev)
    if res is not None:
        return _to_json_safe(res)
    # dataclass
    try:
        if dataclasses.is_dataclass(ev):
            return _to_json_safe(dataclasses.asdict(ev))
    except Exception:
        pass
    # pydantic
    for meth in ("model_dump", "dict"):
        try:
            if hasattr(ev, meth) and callable(getattr(ev, meth)):
                d = getattr(ev, meth)()
                if isinstance(d, dict):
                    return _to_json_safe(d)
        except Exception:
            pass
    # plain object: filter public attrs
    try:
        d = getattr(ev, "__dict__", None)
        if isinstance(d, dict):
            pub = {k: v for k, v in d.items() if not str(k).startswith("_")}
            if pub:
                return _to_json_safe(pub)
    except Exception:
        pass
    # fallback string
    return _safe_str(ev)


def _extract_stop_result(obj: Any) -> Optional[Any]:
    """Extract StopEvent.result safely without invoking callables.

    Supports common patterns:
    - Objects with non-callable `result` attribute.
    - dict/dataclass/pydantic with one of keys: result, value, output, data, payload, content.
    """
    try:
        # If this is a WorkflowHandler (asyncio.Future), only read result when done.
        if isinstance(obj, asyncio.Future) or (
            hasattr(obj, "done") and hasattr(obj, "result")
        ):
            try:
                if obj.done():
                    return obj.result()
                else:
                    return None
            except Exception:
                return None
        if hasattr(obj, "result"):
            res = getattr(obj, "result")
            if not callable(res):
                return res
        # Common key names for output
        keys = ("result", "value", "output", "data", "payload", "content")
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    return obj.get(k)
        # dataclass: try asdict and read 'result'
        if dataclasses.is_dataclass(obj):
            try:
                d = dataclasses.asdict(obj)
                for k in keys:
                    if k in d:
                        return d[k]
            except Exception:
                pass
        # pydantic model: try model_dump/dict
        for meth in ("model_dump", "dict"):
            try:
                if hasattr(obj, meth) and callable(getattr(obj, meth)):
                    d = getattr(obj, meth)()
                    if isinstance(d, dict):
                        for k in keys:
                            if k in d:
                                return d[k]
            except Exception:
                pass
        # Generic object: check __dict__
        try:
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                for k in keys:
                    if k in d:
                        return d[k]
        except Exception:
            pass
    except Exception:
        pass
    return None


class TruLlamaWorkflow(core_app.App):
    """Recorder for LlamaIndex Workflows.

    Automatically instruments only the class methods decorated with @step on
    the wrapped workflow class (equivalent to manual instrument_method per
    step). This avoids extra executor-level spans and yields a 1:1 mapping
    with user-defined steps.
    """

    app: Any
    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)

    _is_instrumented: ClassVar[bool] = False

    @classmethod
    def _ensure_workflows_instrumentation(cls) -> None:
        cls._is_instrumented = True

    def __init__(
        self,
        app: Any,
        main_method: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        # Ensure class-level instrumentation for workflow steps.
        self._ensure_workflows_instrumentation()

        # Store the original workflow
        self._original_workflow = app

        # Wrap the workflow's run method to await completion
        import functools

        original_run = app.run

        @functools.wraps(original_run)
        async def wrapped_run(**run_kwargs):
            """Wrapped run that awaits the workflow handler to get the actual result."""
            # Call the original run (which is sync and returns a handler)
            handler = original_run(**run_kwargs)
            # The handler is a WorkflowHandler which is awaitable
            # Await it to get the actual result and ensure all steps complete
            if hasattr(handler, "__await__"):
                result = await handler
                return result
            return handler

        # Replace the run method with our wrapped version
        app.run = wrapped_run

        kwargs["app"] = app
        # Create or adopt session.
        if "connector" in kwargs:
            TruSession(connector=kwargs["connector"])
        else:
            TruSession()

            # Instrument only attributes that are instances of Step classes.
        cls = type(app)
        for name, val in cls.__dict__.items():
            # Only instrument coroutine step methods: async def with 'ev' parameter
            if not inspect.isfunction(val) or not inspect.iscoroutinefunction(
                val
            ):
                continue
            try:
                params = list(
                    (inspect.signature(val).parameters or {}).values()
                )
                if len(params) < 2 or params[1].name != "ev":
                    continue
            except Exception:
                continue

            def step_function_attributes(ret, exception, *args, **kwargs):  # noqa: ANN001
                # Try to locate `ev`
                ev = None
                try:
                    if isinstance(kwargs, dict) and "ev" in kwargs:
                        ev = kwargs.get("ev")
                    elif len(args) > 1:
                        ev = args[1]
                except Exception:
                    pass

                # Shallow serialize ev
                ev_ser = _serialize_event(ev)

                out_val = _extract_stop_result(ret)
                out_ser = (
                    _to_json_safe(out_val)
                    if out_val is not None
                    else _serialize_event(ret)
                )

                exception_str = _safe_str(exception)

                attrs = {
                    SpanAttributes.WORKFLOW.INPUT_EVENT: ev_ser,
                    SpanAttributes.WORKFLOW.OUTPUT_EVENT: out_ser,
                    SpanAttributes.WORKFLOW.ERROR: exception_str,
                    # Also set generic call attributes so UI input/output panels are populated
                    f"{SpanAttributes.CALL.KWARGS}.ev": ev_ser,
                    SpanAttributes.CALL.RETURN: out_ser,
                }

                return attrs

            try:
                instrument_method(
                    cls=cls,
                    method_name=name,
                    span_type=SpanAttributes.SpanType.WORKFLOW_STEP,
                    attributes=step_function_attributes,
                )
            except Exception as e:
                logger.debug("Could not instrument step method %s: %s", name, e)
        if main_method is not None:
            kwargs["main_method"] = main_method
        kwargs["root_class"] = pyschema_utils.Class.of_object(app)

        super().__init__(**kwargs)


TruLlamaWorkflow.model_rebuild()


# Avoid calling __str__ on workflow handler returns in record-root output.
# Returning None here prevents set_record_root_span_attributes from forcing
# stringification of pending handlers.
def _main_output_override(self, func: Callable, sig, bindings, ret):  # type: ignore[override]
    try:
        # If this is a finished WorkflowHandler, return its result.
        try:
            import asyncio

            if isinstance(ret, asyncio.Future) or (
                hasattr(ret, "done") and hasattr(ret, "result")
            ):
                if ret.done():
                    return _safe_str(ret.result())
        except Exception:
            pass

        # If this is a StopEvent or dict-like StopEvent, extract the stored result.
        val = _extract_stop_result(ret)
        if val is not None:
            return _safe_str(val)

        # Otherwise, ret is likely the already-awaited final value (string/dict/etc.).
        if ret is not None:
            return _safe_str(ret)
    except Exception:
        pass
    return None


TruLlamaWorkflow.main_output = _main_output_override  # type: ignore[assignment]


# Provide a clean main input for workflow.run so record root INPUT is informative.
def _main_input_override(self, func: Callable, sig, bindings):  # type: ignore[override]
    try:
        # Prefer explicit kwargs passed to run (e.g., {'topic': 'pirates'}).
        if isinstance(getattr(bindings, "arguments", None), dict):
            args_map = bindings.arguments
            if "kwargs" in args_map and isinstance(args_map["kwargs"], dict):
                return str(dict(args_map["kwargs"]))
            # Fallback: all non-self bound arguments.
            fallback = {k: v for k, v in args_map.items() if k != "self"}
            if fallback:
                return str(fallback)
    except Exception:
        pass
    return "{}"


TruLlamaWorkflow.main_input = _main_input_override  # type: ignore[assignment]
