import inspect
import logging
import types
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from opentelemetry import trace
from opentelemetry.baggage import get_baggage
from opentelemetry.baggage import remove_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from opentelemetry.trace.span import Span
from trulens.core.otel.function_call_context_manager import (
    create_function_call_context_manager,
)
from trulens.experimental.otel_tracing.core.session import TRULENS_SERVICE_NAME
from trulens.experimental.otel_tracing.core.span import Attributes
from trulens.experimental.otel_tracing.core.span import (
    set_function_call_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_general_span_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_record_root_span_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_user_defined_attributes,
)
from trulens.otel.semconv.constants import TRULENS_INSTRUMENT_WRAPPER_FLAG
from trulens.otel.semconv.constants import (
    TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG,
)
from trulens.otel.semconv.trace import SpanAttributes
import wrapt

logger = logging.getLogger(__name__)


def _get_func_name(func: Callable) -> str:
    if (
        hasattr(func, "__module__")
        and func.__module__
        and hasattr(func, "__qualname__")
        and func.__qualname__
    ):
        return f"{func.__module__}.{func.__qualname__}"
    elif hasattr(func, "__qualname__") and func.__qualname__:
        return func.__qualname__
    else:
        return func.__name__


def _resolve_attributes(
    attributes: Attributes,
    ret: Optional[Any],
    exception: Optional[Exception],
    args: Sequence[Any],
    all_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if attributes is None:
        return {}
    if callable(attributes):
        return attributes(ret, exception, *args, **all_kwargs)
    if isinstance(attributes, dict):
        resolved = {}
        value_string_to_value = all_kwargs.copy()
        value_string_to_value["return"] = ret
        for k, v in attributes.items():
            resolved[k] = value_string_to_value[v]
        return resolved
    return attributes.copy()


def _set_span_attributes(
    span: Span,
    span_type: SpanAttributes.SpanType,
    span_name: str,
    func: Callable,
    func_exception: Optional[Exception],
    attributes: Attributes,
    instance: Any,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    ret: Any,
    only_set_user_defined_attributes: bool = False,
):
    # Determine args/kwargs to pass to the attributes
    # callable.
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs).arguments
    all_kwargs = {**kwargs, **bound_args}
    if not only_set_user_defined_attributes:
        # Set general span attributes.
        span.set_attribute("name", span_name)
        set_general_span_attributes(span, span_type)
        # Set record root span attributes if necessary.
        if span_type == SpanAttributes.SpanType.RECORD_ROOT:
            set_record_root_span_attributes(
                span,
                func,
                args,
                kwargs,
                ret,
                func_exception,
            )
        # Set function call attributes.
        set_function_call_attributes(span, ret, func_exception, all_kwargs)
    # Resolve the attributes.
    if instance is not None:
        args_with_self_possibly = (instance,) + args
    else:
        args_with_self_possibly = args
    resolved_attributes = _resolve_attributes(
        attributes,
        ret,
        func_exception,
        args_with_self_possibly,
        all_kwargs,
    )
    if resolved_attributes:
        # Set the user-provided attributes.
        set_user_defined_attributes(
            span,
            span_type=span_type,
            attributes=resolved_attributes,
        )


def instrument(
    *,
    span_type: SpanAttributes.SpanType = SpanAttributes.SpanType.UNKNOWN,
    attributes: Attributes = None,
    must_be_first_wrapper: bool = False,
    **kwargs,
) -> Callable[[Callable], Any]:  # TODO(this_pr): get type!
    """
    Decorator for marking functions to be instrumented with OpenTelemetry
    tracing.

    span_type: Span type to be used for the span.
    attributes:
        A dictionary or a callable that returns a dictionary of attributes
        (i.e. a `typing.Dict[str, typing.Any]`) to be set on the span.
    must_be_first_wrapper:
        If this is True and the function is already wrapped with the TruLens
        decorator, then the function will not be wrapped again.
    """
    if attributes is None:
        attributes = {}
    is_record_root = span_type == SpanAttributes.SpanType.RECORD_ROOT
    is_app_specific_record_root = kwargs.get(
        "is_app_specific_record_root", False
    )
    create_new_span = kwargs.get("create_new_span", True)
    only_set_user_defined_attributes = kwargs.get(
        "only_set_user_defined_attributes", False
    )

    def inner_decorator(func: Callable):
        span_name = _get_func_name(func)

        @wrapt.decorator
        def sync_wrapper(func, instance, args, kwargs):
            ret = convert_to_generator(func, instance, args, kwargs)
            if next(ret) == "is_not_generator":
                res = next(ret)
                # Check that there are no more entries in the generator.
                valid = False
                try:
                    next(ret)
                except StopIteration:
                    valid = True
                if not valid:
                    raise ValueError("The generator is not empty!")
                ret = res
            return ret

        def convert_to_generator(func, instance, args, kwargs):
            with create_function_call_context_manager(
                create_new_span, span_name, is_record_root
            ) as span:
                ret = None
                func_exception: Optional[Exception] = None
                attributes_exception: Optional[Exception] = None
                # Run function.
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, types.GeneratorType):
                        yield "is_generator"
                        ret = []
                        for curr in result:
                            ret.append(curr)
                            yield curr
                    else:
                        yield "is_not_generator"
                        ret = result
                        yield ret
                except Exception as e:
                    # We want to get into the next clause to allow the users
                    # to still add attributes. It's on the user to deal with
                    # None as a return value.
                    func_exception = e
                finally:
                    # Set span attributes.
                    try:
                        _set_span_attributes(
                            span,
                            span_type,
                            span_name,
                            func,
                            func_exception,
                            attributes,
                            instance,
                            args,
                            kwargs,
                            ret,
                            only_set_user_defined_attributes=only_set_user_defined_attributes,
                        )
                    except Exception as e:
                        logger.error(f"Error setting attributes: {e}")
                        attributes_exception = e
                    # Raise any exceptions that occurred.
                    exception = func_exception or attributes_exception
                    if exception:
                        raise exception
                    return ret

        @wrapt.decorator
        async def async_wrapper(func, instance, args, kwargs):
            with create_function_call_context_manager(
                create_new_span, span_name, is_record_root
            ) as span:
                ret = None
                func_exception: Optional[Exception] = None
                attributes_exception: Optional[Exception] = None
                # Run function.
                try:
                    ret = await func(*args, **kwargs)
                except Exception as e:
                    # We want to get into the next clause to allow the users
                    # to still add attributes. It's on the user to deal with
                    # None as a return value.
                    func_exception = e
                # Set span attributes.
                try:
                    _set_span_attributes(
                        span,
                        span_type,
                        span_name,
                        func,
                        func_exception,
                        attributes,
                        instance,
                        args,
                        kwargs,
                        ret,
                    )
                except Exception as e:
                    logger.error(f"Error setting attributes: {e}")
                    attributes_exception = e
                # Raise any exceptions that occurred.
                exception = func_exception or attributes_exception
                if exception:
                    raise exception
            return ret

        @wrapt.decorator
        async def async_generator_wrapper(func, instance, args, kwargs):
            with create_function_call_context_manager(
                create_new_span, span_name, is_record_root
            ) as span:
                ret = None
                func_exception: Optional[Exception] = None
                attributes_exception: Optional[Exception] = None
                # Run function.
                try:
                    result = func(*args, **kwargs)
                    ret = []
                    async for curr in result:
                        ret.append(curr)
                        yield curr
                except Exception as e:
                    # We want to get into the next clause to allow the users
                    # to still add attributes. It's on the user to deal with
                    # None as a return value.
                    func_exception = e
                finally:
                    # Set span attributes.
                    try:
                        _set_span_attributes(
                            span,
                            span_type,
                            span_name,
                            func,
                            func_exception,
                            attributes,
                            instance,
                            args,
                            kwargs,
                            ret,
                        )
                    except Exception as e:
                        logger.error(f"Error setting attributes: {e}")
                        attributes_exception = e
                    # Raise any exceptions that occurred.
                    exception = func_exception or attributes_exception
                    if exception:
                        raise exception

        # Check if already wrapped if not allowing multiple wrappers.
        if must_be_first_wrapper:
            if hasattr(func, TRULENS_INSTRUMENT_WRAPPER_FLAG):
                return func
            curr = func
            while hasattr(curr, "__wrapped__"):
                curr = curr.__wrapped__
                if hasattr(curr, TRULENS_INSTRUMENT_WRAPPER_FLAG):
                    return func

        # Wrap.
        ret = None
        if inspect.isasyncgenfunction(func):
            ret = async_generator_wrapper(func)
        elif inspect.iscoroutinefunction(func):
            ret = async_wrapper(func)
        else:
            ret = sync_wrapper(func)
        ret.__dict__[TRULENS_INSTRUMENT_WRAPPER_FLAG] = True
        if is_record_root and not is_app_specific_record_root:
            ret.__dict__[TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG] = True
        return ret

    return inner_decorator


def instrument_method(
    cls: type,
    method_name: str,
    span_type: SpanAttributes.SpanType = SpanAttributes.SpanType.UNKNOWN,
    attributes: Attributes = None,
    must_be_first_wrapper: bool = False,
) -> None:
    wrapper = instrument(
        span_type=span_type,
        attributes=attributes,
        must_be_first_wrapper=must_be_first_wrapper,
    )
    setattr(cls, method_name, wrapper(getattr(cls, method_name)))


def instrument_cost_computer(
    cls: type,
    method_name: str,
    attributes: Attributes,
) -> None:
    wrapper = instrument(
        attributes=attributes,
        create_new_span=False,
        only_set_user_defined_attributes=True,
    )
    setattr(cls, method_name, wrapper(getattr(cls, method_name)))


class OtelBaseRecordingContext:
    run_name: str
    """
    The name of the run that the recording context is currently processing.
    """

    input_id: str
    """
    The ID of the input that the recording context is currently processing.
    """

    tokens: List[object] = []
    """
    OTEL context tokens for the current context manager. These tokens are how the OTEL
    context api keeps track of what is changed in the context, and used to undo the changes.
    """

    context_keys_added: List[str] = []
    """
    Keys added to the OTEL context.
    """

    def __init__(
        self, *, app_name: str, app_version: str, run_name: str, input_id: str
    ):
        self.app_name = app_name
        self.app_version = app_version
        self.run_name = run_name
        self.input_id = input_id
        self.tokens = []
        self.context_keys_added = []
        self.span_context = None

    # Calling set_baggage does not actually add the baggage to the current context, but returns a new one
    # To avoid issues with remembering to add/remove the baggage, we attach it to the runtime context.
    def attach_to_context(self, key: str, value: object):
        if get_baggage(key) or value is None:
            return

        self.tokens.append(context_api.attach(set_baggage(key, value)))
        self.context_keys_added.append(key)

    async def __aenter__(self):
        return self.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Exiting the span context before updating the context to ensure nothing
        # carries over unintentionally
        if self.span_context:
            # TODO[SNOW-1854360]: Add in feature function spans.
            self.span_context.__exit__(exc_type, exc_value, exc_tb)

        logger.debug("Exiting the OTEL app context.")

        # Clearing the context / baggage added.
        while self.context_keys_added:
            remove_baggage(self.context_keys_added.pop())

        while self.tokens:
            # Clearing the context once we're done with this root span.
            # See https://github.com/open-telemetry/opentelemetry-python/issues/2432#issuecomment-1593458684
            context_api.detach(self.tokens.pop())

    async def __aexit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        return self.__exit__(exc_type, exc_val, exc_tb)


class OtelRecordingContext(OtelBaseRecordingContext):
    def __init__(
        self,
        *,
        app_name: str,
        app_version: str,
        run_name: str,
        input_id: str,
        ground_truth_output: Optional[str] = None,
    ) -> None:
        super().__init__(
            app_name=app_name,
            app_version=app_version,
            run_name=run_name,
            input_id=input_id,
        )
        self.ground_truth_output = ground_truth_output

    # For use as a context manager.
    def __enter__(self) -> None:
        self.attach_to_context(SpanAttributes.APP_NAME, self.app_name)
        self.attach_to_context(SpanAttributes.APP_VERSION, self.app_version)

        self.attach_to_context(SpanAttributes.RUN_NAME, self.run_name)
        self.attach_to_context(SpanAttributes.INPUT_ID, self.input_id)
        self.attach_to_context(
            SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
            self.ground_truth_output,
        )


class OtelFeedbackComputationRecordingContext(OtelBaseRecordingContext):
    def __init__(self, *args, **kwargs):
        self.target_record_id = kwargs.pop("target_record_id")
        self.feedback_name = kwargs.pop("feedback_name")
        super().__init__(*args, **kwargs)

    # For use as a context manager.
    def __enter__(self) -> Span:
        tracer = trace.get_tracer_provider().get_tracer(TRULENS_SERVICE_NAME)

        self.attach_to_context(
            SpanAttributes.RECORD_ID, self.target_record_id
        )  # TODO(otel): Should we include this? It's automatically getting added to the span.
        self.attach_to_context(SpanAttributes.APP_NAME, self.app_name)
        self.attach_to_context(SpanAttributes.APP_VERSION, self.app_version)

        self.attach_to_context(SpanAttributes.RUN_NAME, self.run_name)
        self.attach_to_context(
            SpanAttributes.EVAL.TARGET_RECORD_ID, self.target_record_id
        )
        self.attach_to_context(SpanAttributes.INPUT_ID, self.input_id)
        self.attach_to_context(
            SpanAttributes.EVAL.METRIC_NAME, self.feedback_name
        )

        # Use start_as_current_span as a context manager
        self.span_context = tracer.start_as_current_span("eval_root")
        root_span = self.span_context.__enter__()
        root_span_id = str(root_span.get_span_context().span_id)

        self.attach_to_context(
            SpanAttributes.EVAL.EVAL_ROOT_ID,
            root_span_id,
        )

        # Set general span attributes
        root_span.set_attribute("name", "eval_root")
        set_general_span_attributes(
            root_span, SpanAttributes.SpanType.EVAL_ROOT
        )

        return root_span
