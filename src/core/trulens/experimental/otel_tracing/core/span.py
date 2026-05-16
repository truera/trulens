"""
This file contains utility functions specific to certain span types.
"""

from __future__ import annotations

from inspect import signature
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from opentelemetry.baggage import get_baggage
from opentelemetry.context import Context
from opentelemetry.trace.span import Span
from opentelemetry.util.types import AttributeValue
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

if TYPE_CHECKING:
    from trulens.core.app import App

logger = logging.getLogger(__name__)

"""
Type definitions
"""
Attributes = Optional[
    Union[
        Dict[str, Any],
        Callable[
            [
                Optional[Any],
                Optional[Exception],
                List[Any],
                Optional[Dict[str, Any]],
            ],
            Dict[str, Any],
        ],
    ]
]

"""
General utilites for all spans
"""


def _stringify_span_attribute(o: Any) -> Tuple[bool, str]:
    """Converts an object to a string.

    Args:
        o: object to convert to a string

    Returns:
        A tuple containing:
        - A boolean indicating whether the object was jsonified (as opposed to
          simple stringification)
        - The string representation of the object.
    """
    try:
        return True, json.dumps(o)
    except Exception:
        pass
    # Special handling for exceptions to preserve their message
    if isinstance(o, Exception):
        try:
            return False, str(o)
        except Exception:
            return False, f"<{type(o).__name__}>"
    # For everything else, try str() but fallback to type name
    # This maintains backward compatibility with tests while avoiding
    # issues with objects that have poor __str__ implementations
    try:
        return False, str(o)
    except Exception:
        # If str() fails, use type name instead
        try:
            return False, f"<{type(o).__name__}>"
        except Exception:
            return False, "<unstringifiable>"


def _convert_to_valid_span_attribute_type(val: Any) -> AttributeValue:
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        for curr_type in [bool, int, float, str]:
            if all([isinstance(curr, curr_type) for curr in val]):
                return val
        jsonifiable, j = _stringify_span_attribute(val)
        if jsonifiable:
            return j
        return [_stringify_span_attribute(curr)[1] for curr in val]
    return _stringify_span_attribute(val)[1]


def set_span_attribute_safely(
    span: Span,
    key: str,
    value: Any,
) -> None:
    if value is not None:
        span.set_attribute(key, _convert_to_valid_span_attribute_type(value))


def set_string_span_attribute_from_baggage(
    span: Span, key: str, context: Optional[Context] = None
) -> None:
    value = get_baggage(key, context)
    if value is not None:
        span.set_attribute(key, str(value))


def validate_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to validate span attributes based on the span type.
    """
    if not isinstance(attributes, dict) or any([
        not isinstance(key, str) for key in attributes.keys()
    ]):
        raise ValueError("Attributes must be a dictionary with string keys.")

    if SpanAttributes.SPAN_TYPE in attributes:
        raise ValueError("Span type should not be set in attributes.")

    # TODO: validate Span type attributes.
    return attributes


def set_general_span_attributes(
    span: Span,
    /,
    span_type: SpanAttributes.SpanType,
    context: Optional[Context] = None,
) -> None:
    span.set_attribute(SpanAttributes.SPAN_TYPE, span_type)

    set_string_span_attribute_from_baggage(
        span, ResourceAttributes.APP_NAME, context
    )
    set_string_span_attribute_from_baggage(
        span, ResourceAttributes.APP_VERSION, context
    )
    set_string_span_attribute_from_baggage(
        span, ResourceAttributes.APP_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.RECORD_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.EVAL.TARGET_RECORD_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.EVAL.EVAL_ROOT_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.EVAL.METRIC_NAME, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.RUN_NAME, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.INPUT_ID, context
    )
    set_string_span_attribute_from_baggage(
        span, SpanAttributes.INPUT_RECORDS_COUNT, context
    )


def set_function_call_attributes(
    span: Span,
    ret: Any,
    func_name: str,
    func_exception: Optional[Exception],
    all_kwargs: Dict[str, Any],
) -> None:
    set_span_attribute_safely(span, SpanAttributes.CALL.RETURN, ret)
    set_span_attribute_safely(span, SpanAttributes.CALL.FUNCTION, func_name)
    set_span_attribute_safely(span, SpanAttributes.CALL.ERROR, func_exception)
    for k, v in all_kwargs.items():
        set_span_attribute_safely(span, f"{SpanAttributes.CALL.KWARGS}.{k}", v)


def set_user_defined_attributes(
    span: Span,
    *,
    attributes: Dict[str, Any],
) -> None:
    final_attributes = validate_attributes(attributes)

    cost_attributes = {}
    for key, value in final_attributes.items():
        if key.startswith(f"{SpanAttributes.COST.base}."):
            cost_attributes[key] = value
        else:
            set_span_attribute_safely(span, key, value)
    if cost_attributes:
        attributes_so_far = dict(getattr(span, "attributes", {}))
        if attributes_so_far:
            for cost_field in [
                SpanAttributes.COST.COST,
                SpanAttributes.COST.NUM_TOKENS,
                SpanAttributes.COST.NUM_PROMPT_TOKENS,
                SpanAttributes.COST.NUM_COMPLETION_TOKENS,
            ]:
                cost_attributes[cost_field] = cost_attributes.get(
                    cost_field, 0
                ) + attributes_so_far.get(cost_field, 0)
            currency = attributes_so_far.get(SpanAttributes.COST.CURRENCY, None)
            model = attributes_so_far.get(SpanAttributes.COST.MODEL, None)
            if currency not in [
                None,
                cost_attributes.get(SpanAttributes.COST.CURRENCY),
            ]:
                cost_attributes[SpanAttributes.COST.CURRENCY] = "mixed"
            if model not in [
                None,
                cost_attributes.get(SpanAttributes.COST.MODEL),
            ]:
                cost_attributes[SpanAttributes.COST.MODEL] = "mixed"
            for k, v in cost_attributes.items():
                set_span_attribute_safely(span, k, v)


"""
RECORD_ROOT SPAN
"""


def set_record_root_span_attributes(
    span: Span,
    /,
    func: Callable,
    args: tuple,
    kwargs: dict,
    ret: Any,
    exception: Optional[Exception],
) -> None:
    tru_app: App = get_baggage("__trulens_app__")
    sig = signature(func)
    input_selector = get_baggage("__trulens_input_selector__")

    # Use custom input selector if provided, otherwise fall back to default
    if input_selector is not None and callable(input_selector):
        try:
            main_input_value = input_selector(args, kwargs)
        except Exception as e:
            logger.warning(
                f"Custom input selector failed: {e}. Falling back to default."
            )
            main_input_value = tru_app.main_input(
                func, sig, sig.bind_partial(*args, **kwargs)
            )
    else:
        main_input_value = tru_app.main_input(
            func, sig, sig.bind_partial(*args, **kwargs)
        )

    set_span_attribute_safely(
        span,
        SpanAttributes.RECORD_ROOT.INPUT,
        main_input_value,
    )
    ground_truth_output = get_baggage(
        SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT
    )
    if ground_truth_output:
        set_span_attribute_safely(
            span,
            SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
            ground_truth_output,
        )
    if exception:
        set_span_attribute_safely(
            span, SpanAttributes.RECORD_ROOT.ERROR, str(exception)
        )
    if ret is not None:
        set_span_attribute_safely(
            span,
            SpanAttributes.RECORD_ROOT.OUTPUT,
            tru_app.main_output(
                func, sig, sig.bind_partial(*args, **kwargs), ret
            ),
        )


"""
GenAI semantic convention dual-emit helpers.

Each helper emits the official ``gen_ai.*`` OTEL attributes alongside
the existing ``ai.observability.*`` attributes so that downstream
collectors that understand the OTEL GenAI spec can consume TruLens data
without requiring TruLens-specific attribute knowledge.
"""


def set_genai_generation_attributes(
    span: Span,
    *,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    provider_name: Optional[str] = None,
    operation_name: Optional[str] = None,
) -> None:
    """Emit ``gen_ai.*`` attributes for a GENERATION span.

    Args:
        span: The OTEL span to annotate.
        model: Name of the model used (maps to
            ``gen_ai.request.model``).
        input_tokens: Number of prompt tokens consumed (maps to
            ``gen_ai.usage.input_tokens``).
        output_tokens: Number of completion tokens generated (maps to
            ``gen_ai.usage.output_tokens``).
        temperature: Sampling temperature (maps to
            ``gen_ai.request.temperature``).
        provider_name: GenAI provider name, e.g. ``"openai"`` (maps to
            ``gen_ai.system``).
        operation_name: GenAI operation name (maps to
            ``gen_ai.operation.name``). When ``None`` the attribute is
            not emitted.
    """
    if operation_name is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.OPERATION.NAME, operation_name
        )
    if model is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.REQUEST.MODEL, model
        )
    if input_tokens is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.USAGE.INPUT_TOKENS, input_tokens
        )
    if output_tokens is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.USAGE.OUTPUT_TOKENS, output_tokens
        )
    if temperature is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.REQUEST.TEMPERATURE, temperature
        )
    if provider_name is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.SYSTEM.NAME, provider_name
        )


def set_genai_retrieval_attributes(
    span: Span,
    *,
    query_text: Optional[str] = None,
    documents: Optional[Any] = None,
) -> None:
    """Emit ``gen_ai.*`` attributes for a RETRIEVAL span.

    Args:
        span: The OTEL span to annotate.
        query_text: Query used for retrieval (maps to
            ``gen_ai.retrieval.query.text``).
        documents: Retrieved documents or contexts (maps to
            ``gen_ai.retrieval.documents``).
    """
    if query_text is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.RETRIEVAL.QUERY_TEXT, query_text
        )
    if documents is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.RETRIEVAL.DOCUMENTS, documents
        )


def set_genai_tool_attributes(
    span: Span,
    *,
    tool_name: Optional[str] = None,
    call_arguments: Optional[Any] = None,
    call_result: Optional[Any] = None,
) -> None:
    """Emit ``gen_ai.*`` attributes for a TOOL or MCP span.

    Args:
        span: The OTEL span to annotate.
        tool_name: Name of the tool (maps to ``gen_ai.tool.name``).
        call_arguments: Arguments passed to the tool (maps to
            ``gen_ai.tool.call.arguments``).
        call_result: Result returned by the tool (maps to
            ``gen_ai.tool.call.result``).
    """
    if tool_name is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.TOOL.NAME, tool_name
        )
    if call_arguments is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.TOOL.CALL_ARGUMENTS, call_arguments
        )
    if call_result is not None:
        set_span_attribute_safely(
            span, GenAIAttributes.TOOL.CALL_RESULT, call_result
        )
