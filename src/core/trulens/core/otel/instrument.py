from __future__ import annotations

import inspect
import logging
import types
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from opentelemetry import trace
from opentelemetry.baggage import get_baggage
from opentelemetry.baggage import remove_baggage
from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from opentelemetry.trace.span import Span
from trulens.core.otel.function_call_context_manager import (
    create_function_call_context_manager,
)
from trulens.core.otel.recording import Recording
from trulens.core.schema.app import AppDefinition
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
from trulens.otel.semconv.constants import (
    TRULENS_APP_SPECIFIC_INSTRUMENT_WRAPPER_FLAG,
)
from trulens.otel.semconv.constants import TRULENS_INSTRUMENT_WRAPPER_FLAG
from trulens.otel.semconv.constants import (
    TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG,
)
from trulens.otel.semconv.constants import TRULENS_SPAN_END_CALLBACKS
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes
import wrapt

if TYPE_CHECKING:
    from trulens.core.app import App


logger = logging.getLogger(__name__)


def get_func_name(func: Callable) -> str:
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
    func_name: str,
    func: Callable,
    func_exception: Optional[Exception],
    attributes: Attributes,
    instance: Any,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    ret: Any,
    only_set_user_defined_attributes: bool = False,
):
    if (
        hasattr(span, "attributes")
        and span.attributes is not None
        and SpanAttributes.SPAN_TYPE in span.attributes
        and span.attributes[SpanAttributes.SPAN_TYPE]
        not in [None, "", SpanAttributes.SpanType.UNKNOWN]
    ):
        # If the span already has a span type, we override what we were given.
        span_type = span.attributes[SpanAttributes.SPAN_TYPE]
    # Determine args/kwargs to pass to the attributes
    # callable.
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs).arguments
    all_kwargs = {**kwargs, **bound_args}
    if not only_set_user_defined_attributes:
        # Set general span attributes.
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
        set_function_call_attributes(
            span, ret, func_name, func_exception, all_kwargs
        )
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
        set_user_defined_attributes(span, attributes=resolved_attributes)


def _finalize_span(
    span: Span,
    span_type: SpanAttributes.SpanType,
    func_name: str,
    func: Callable,
    func_exception: Optional[Exception],
    attributes: Attributes,
    instance: Any,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    ret: Any,
    only_set_user_defined_attributes: bool = False,
    span_end_callbacks: List[Callable[[Span], None]] = [],
):
    attributes_exception: Optional[Exception] = None
    try:
        _set_span_attributes(
            span,
            span_type,
            func_name,
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
    for span_end_callback in span_end_callbacks:
        span_end_callback(span)
    # Raise any exceptions that occurred.
    exception = func_exception or attributes_exception
    if exception:
        raise exception


class instrument:
    enabled: bool = True

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        span_type: SpanAttributes.SpanType = SpanAttributes.SpanType.UNKNOWN,
        attributes: Attributes = None,
        **kwargs,
    ) -> None:
        """
        Decorator for marking functions to be instrumented with OpenTelemetry
        tracing.

        name: Optional custom span name. If not provided, derives from function's
            __module__.__qualname__. Use this for clean span names without
            module prefixes (e.g., name="call_llm" instead of "__main__.call_llm").
        span_type: Span type to be used for the span.
        attributes:
            A dictionary or a callable that returns a dictionary of attributes
            (i.e. a `typing.Dict[str, typing.Any]`) to be set on the span.
        """
        self.name = name
        self.user_specified_span_type = span_type
        self.span_type = span_type
        if span_type == SpanAttributes.SpanType.RECORD_ROOT:
            logger.warning(
                "Cannot explicitly set 'record_root' span type, setting to 'unknown'."
            )
            self.span_type = SpanAttributes.SpanType.UNKNOWN
        if attributes is None:
            attributes = {}
        self.attributes = attributes
        self.is_app_specific_instrumentation = kwargs.get(
            "is_app_specific_instrumentation", False
        )
        self.create_new_span = kwargs.get("create_new_span", True)
        self.only_set_user_defined_attributes = kwargs.get(
            "only_set_user_defined_attributes", False
        )
        self.must_be_first_wrapper = kwargs.get("must_be_first_wrapper", False)

    def __call__(self, func: Callable) -> Callable:
        func_name = self.name if self.name else get_func_name(func)
        custom_name_provided = self.name is not None

        def _func_name_for_instance(instance) -> str:
            # If a custom name was explicitly provided, always use it
            if custom_name_provided:
                return func_name
            target = None
            if instance is None:
                target = None
            elif inspect.isclass(instance):
                target = instance
            else:
                target = instance.__class__
            if target is None:
                return func_name
            module = getattr(target, "__module__", "")
            qualname = getattr(
                target, "__qualname__", getattr(target, "__name__", "")
            )
            base = ".".join([p for p in [module, qualname] if p])
            if not base:
                return func_name
            return f"{base}.{func.__name__}"

        @wrapt.decorator
        def sync_wrapper(func, instance, args, kwargs):
            if not self.enabled or get_baggage("__trulens_recording__") is None:
                return func(*args, **kwargs)
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
            span_end_callbacks = kwargs.pop(TRULENS_SPAN_END_CALLBACKS, [])
            func_name_for_call = _func_name_for_instance(instance)
            with create_function_call_context_manager(
                self.create_new_span, func_name_for_call
            ) as span:
                ret = None
                func_exception: Optional[Exception] = None
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
                    _finalize_span(
                        span,
                        self.span_type,
                        func_name_for_call,
                        func,
                        func_exception,
                        self.attributes,
                        instance,
                        args,
                        kwargs,
                        ret,
                        self.only_set_user_defined_attributes,
                        span_end_callbacks,
                    )
                    return ret

        @wrapt.decorator
        async def async_wrapper(func, instance, args, kwargs):
            if not self.enabled or get_baggage("__trulens_recording__") is None:
                return await func(*args, **kwargs)
            span_end_callbacks = kwargs.pop(TRULENS_SPAN_END_CALLBACKS, [])
            func_name_for_call = _func_name_for_instance(instance)
            with create_function_call_context_manager(
                self.create_new_span, func_name_for_call
            ) as span:
                ret = None
                func_exception: Optional[Exception] = None
                # Run function.
                try:
                    ret = await func(*args, **kwargs)
                except Exception as e:
                    # We want to get into the next clause to allow the users
                    # to still add attributes. It's on the user to deal with
                    # None as a return value.
                    func_exception = e
                # Set span attributes.
                _finalize_span(
                    span,
                    self.span_type,
                    func_name_for_call,
                    func,
                    func_exception,
                    self.attributes,
                    instance,
                    args,
                    kwargs,
                    ret,
                    self.only_set_user_defined_attributes,
                    span_end_callbacks,
                )
            return ret

        @wrapt.decorator
        async def async_generator_wrapper(func, instance, args, kwargs):
            if not self.enabled or get_baggage("__trulens_recording__") is None:
                async for curr in func(*args, **kwargs):
                    yield curr
                return
            span_end_callbacks = kwargs.pop(TRULENS_SPAN_END_CALLBACKS, [])
            func_name_for_call = _func_name_for_instance(instance)
            with create_function_call_context_manager(
                self.create_new_span, func_name_for_call
            ) as span:
                ret = None
                func_exception: Optional[Exception] = None
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
                    _finalize_span(
                        span,
                        self.span_type,
                        func_name_for_call,
                        func,
                        func_exception,
                        self.attributes,
                        instance,
                        args,
                        kwargs,
                        ret,
                        self.only_set_user_defined_attributes,
                        span_end_callbacks,
                    )

        # Check if already wrapped if not allowing multiple wrappers.
        if self.must_be_first_wrapper:
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
        if self.user_specified_span_type == SpanAttributes.SpanType.RECORD_ROOT:
            ret.__dict__[TRULENS_RECORD_ROOT_INSTRUMENT_WRAPPER_FLAG] = True
        if self.is_app_specific_instrumentation:
            ret.__dict__[TRULENS_APP_SPECIFIC_INSTRUMENT_WRAPPER_FLAG] = True
        return ret

    @classmethod
    def enable_all_instrumentation(cls) -> None:
        cls.enabled = True

    @classmethod
    def disable_all_instrumentation(cls) -> None:
        cls.enabled = False


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
        self,
        *,
        app_name: str,
        app_version: str,
        app_id: str,
        run_name: str,
        input_id: str,
    ):
        self.app_name = app_name
        self.app_version = app_version
        self.app_id = app_id
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
        tru_app: App,
        app_name: str,
        app_version: str,
        run_name: str,
        input_id: str,
        input_records_count: Optional[int] = None,
        ground_truth_output: Optional[str] = None,
        input_selector: Optional[
            Callable[[Tuple[Any, ...], Dict[str, Any]], Any]
        ] = None,
    ) -> None:
        app_id = AppDefinition._compute_app_id(app_name, app_version)
        super().__init__(
            app_name=app_name,
            app_version=app_version,
            app_id=app_id,
            run_name=run_name,
            input_id=input_id,
        )
        self.tru_app = tru_app
        self.input_records_count = input_records_count
        self.ground_truth_output = ground_truth_output
        self.input_selector = input_selector

    # For use as a context manager.
    def __enter__(self) -> Recording:
        self.attach_to_context("__trulens_app__", self.tru_app)
        self.attach_to_context(ResourceAttributes.APP_NAME, self.app_name)
        self.attach_to_context(ResourceAttributes.APP_VERSION, self.app_version)
        self.attach_to_context(ResourceAttributes.APP_ID, self.app_id)
        # Attach self so child spans can register baggage tokens for cleanup at app exit.
        self.attach_to_context("__trulens_otel_ctx__", self)
        self.attach_to_context(SpanAttributes.RUN_NAME, self.run_name)
        self.attach_to_context(SpanAttributes.INPUT_ID, self.input_id)

        self.attach_to_context(
            SpanAttributes.INPUT_RECORDS_COUNT,
            self.input_records_count,
        )
        self.attach_to_context(
            SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
            self.ground_truth_output,
        )
        self.attach_to_context(
            "__trulens_input_selector__", self.input_selector
        )

        ret = Recording(self.tru_app)
        self.attach_to_context("__trulens_recording__", ret)
        return ret


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
        self.attach_to_context(ResourceAttributes.APP_NAME, self.app_name)
        self.attach_to_context(ResourceAttributes.APP_VERSION, self.app_version)
        self.attach_to_context(ResourceAttributes.APP_ID, self.app_id)

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
        set_general_span_attributes(
            root_span, SpanAttributes.SpanType.EVAL_ROOT
        )
        root_span.set_attribute(
            SpanAttributes.EVAL_ROOT.METRIC_NAME, self.feedback_name
        )

        self.attach_to_context("__trulens_recording__", True)

        return root_span


# === Trace Beautification Utilities ===


def extract_input_content(messages) -> str:
    """Extract the text content from the input messages.

    Looks for the last HumanMessage's content, or falls back to the first
    message's content if no HumanMessage is found.

    Args:
        messages: List of message objects (e.g., LangChain messages)

    Returns:
        The extracted text content as a string
    """
    if not messages:
        return ""

    # Try to find the last human message
    for msg in reversed(messages):
        # Check for LangChain HumanMessage
        if hasattr(msg, "content"):
            msg_type = type(msg).__name__
            if msg_type == "HumanMessage":
                return msg.content

    # Fall back to first message content
    if hasattr(messages[0], "content"):
        return messages[0].content

    return str(messages[0])


def extract_output_content(ret) -> str:
    """Extract the text content from an LLM response.

    Args:
        ret: The LLM response object

    Returns:
        The extracted text content as a string
    """
    if ret is None:
        return ""
    if hasattr(ret, "content"):
        return ret.content
    return str(ret)


def extract_tool_calls(ret) -> Optional[str]:
    """Extract and format tool calls from an LLM response.

    Formats tool calls as: "tool_name(arg1=val1, arg2=val2), other_tool(...)"

    Args:
        ret: The LLM response object

    Returns:
        Formatted string of tool calls, or None if no tool calls
    """
    if ret is None:
        return None
    if not hasattr(ret, "tool_calls") or not ret.tool_calls:
        return None

    formatted = []
    for tc in ret.tool_calls:
        name = (
            tc.get("name", "unknown")
            if isinstance(tc, dict)
            else getattr(tc, "name", "unknown")
        )
        args = (
            tc.get("args", {})
            if isinstance(tc, dict)
            else getattr(tc, "args", {})
        )

        if isinstance(args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        else:
            args_str = str(args)

        formatted.append(f"{name}({args_str})")

    return ", ".join(formatted)


def generation_attributes() -> Callable:
    """Create an attributes lambda for GENERATION spans.

    Extracts input_content, output_content, and tool_calls from the function
    call and return value.

    Returns:
        A callable suitable for the @instrument(attributes=...) parameter

    Example:
        @instrument(
            name="call_llm",
            span_type=SpanAttributes.SpanType.GENERATION,
            attributes=generation_attributes()
        )
        def call_llm(messages):
            return model.invoke(messages)
    """

    def _extract(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        result = {}

        # Extract input content from first positional arg (usually messages)
        if args:
            input_content = extract_input_content(args[0])
            if input_content:
                result["input_content"] = input_content

        # Extract output content and tool calls from return value
        if ret is not None:
            output_content = extract_output_content(ret)
            if output_content:
                result["output_content"] = output_content

            tool_calls = extract_tool_calls(ret)
            if tool_calls:
                result["tool_calls"] = tool_calls

        return result

    return _extract


def instrument_tools(
    tools_by_name: Dict[str, Any],
    *,
    invoke_method: str = "invoke",
) -> None:
    """Instrument a tools dictionary in place for clean tool span names.

    Replaces each tool in the dictionary with a wrapper that produces spans
    named after the tool (e.g., "add", "multiply") when invoke() is called.

    This is the least invasive way to get clean tool spans - no changes
    to app code required beyond this one setup call.

    Args:
        tools_by_name: Dictionary mapping tool names to tool objects
        invoke_method: Name of the method to wrap (default: "invoke")

    Example:
        tools_by_name = {"add": add_tool, "multiply": multiply_tool}

        # One line setup - wraps tools in place
        instrument_tools(tools_by_name)

        # App code unchanged - just call tool.invoke() as normal
        def call_tool(tool_call):
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])  # Now creates "add" span
            return ToolMessage(content=result, ...)
    """
    for name, tool in list(tools_by_name.items()):
        original_invoke = getattr(tool, invoke_method)

        # Create instrumented invoke with clean span name
        instrumented_invoke = instrument(
            name=name,
            span_type=SpanAttributes.SpanType.TOOL,
        )(original_invoke)

        # Create a wrapper that delegates to the original tool
        # but uses the instrumented invoke
        class ToolWrapper:
            def __init__(self, original_tool, instrumented_invoke):
                self._tool = original_tool
                self._instrumented_invoke = instrumented_invoke

            def invoke(self, *args, **kwargs):
                return self._instrumented_invoke(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._tool, name)

        tools_by_name[name] = ToolWrapper(tool, instrumented_invoke)
