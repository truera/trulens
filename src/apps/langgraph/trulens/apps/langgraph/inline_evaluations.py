import inspect
from typing import Any, Callable

from langchain_core.messages import SystemMessage
from opentelemetry.trace.span import Span
from trulens.core.feedback import Feedback
from trulens.feedback.computer import _call_feedback_function
from trulens.otel.semconv.constants import TRULENS_SPAN_END_CALLBACKS
from trulens.otel.semconv.trace import SpanAttributes
import wrapt


class inline_evaluation:
    def __init__(self, feedback: Feedback, emit_spans: bool = True) -> None:
        self._feedback = feedback
        self._emit_spans = emit_spans

    def __call__(self, func: Callable) -> Callable:
        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            def span_end_callback(span: Span) -> None:
                # Get span attributes
                span_attributes = (
                    dict(span.attributes) if span.attributes else {}
                )
                span_name = getattr(span, "name", None)

                # Try to extract feedback arguments from this span
                feedback_function_inputs = {}
                feedback_args = {}
                for arg_name, selector in self._feedback.selectors.items():
                    if selector.matches_span(span_name, span_attributes):
                        feedback_input = selector.process_span(
                            str(span.get_span_context().span_id),
                            span_attributes,
                        )
                        if (
                            feedback_input.value is not None
                            or not selector.ignore_none_values
                        ):
                            feedback_function_inputs[arg_name] = feedback_input
                            feedback_args[arg_name] = feedback_input.value
                        else:
                            raise ValueError(
                                f"Argument {arg_name} could not be determined!"
                            )
                    else:
                        raise ValueError(
                            f"Selector {selector} did not match span "
                            f"{span_name} with attributes {span_attributes}"
                        )

                if self._emit_spans:
                    # Get necessary attributes from the original span.
                    app_name = span_attributes.get(SpanAttributes.RUN_NAME, "")
                    app_version = span_attributes.get("app_version", "")
                    app_id = span_attributes.get("app_id", "")
                    run_name = span_attributes.get(SpanAttributes.RUN_NAME, "")
                    input_id = span_attributes.get(SpanAttributes.INPUT_ID, "")
                    target_record_id = span_attributes.get(
                        SpanAttributes.RECORD_ID, ""
                    )
                    feedback_result = _call_feedback_function(
                        self._feedback.name,
                        self._feedback.imp,
                        self._feedback.higher_is_better,
                        self._feedback.aggregator,
                        feedback_function_inputs,
                        app_name,
                        app_version,
                        app_id,
                        run_name,
                        input_id,
                        target_record_id,
                        None,
                    )
                else:
                    # Call the feedback function without creating spans
                    feedback_result = self._feedback.imp(**feedback_args)

                # Add feedback result to state messages
                state = self._get_state_arg(func, instance, args, kwargs)
                state["messages"].append(SystemMessage(str(feedback_result)))

            kwargs_copy = kwargs.copy()
            span_callbacks = kwargs_copy.pop(TRULENS_SPAN_END_CALLBACKS, [])
            span_callbacks.append(span_end_callback)
            kwargs_copy[TRULENS_SPAN_END_CALLBACKS] = span_callbacks

            # Execute the original function first
            return func(*args, **kwargs_copy)

        return wrapper(func)

    @staticmethod
    def _get_state_arg(func, instance, args, kwargs) -> Any:
        """
        Get the state argument from the function signature. This assumes that
        it's the first non-instance parameter.

        Args:
            func: The function to get the state argument from.
            instance: The instance of the class that the function is called on.
            args: The positional arguments to the function.
            kwargs: The keyword arguments to the function.

        Returns:
            Any: The state argument.
        """
        # Extract state from the first non-instance parameter
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        # Get parameter names, skipping 'self' if it exists
        param_names = list(sig.parameters.keys())
        if (
            instance is not None
            and param_names
            and param_names[0] in ["self", "cls"]
        ):
            first_param_name = param_names[1] if len(param_names) > 1 else None
        else:
            first_param_name = param_names[0] if param_names else None

        if not first_param_name:
            raise ValueError(
                "Could not find state argument in function signature!"
            )

        return bound_args.arguments.get(first_param_name)
