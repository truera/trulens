import inspect
from typing import Any, Callable

from opentelemetry.trace.span import Span
from trulens.core.feedback import Feedback
from trulens.otel.semconv.trulens.otel.semconv.constants import (
    TRULENS_SPAN_END_CALLBACKS,
)
import wrapt


class inline_evaluation:
    def __init__(self, feedback: Feedback) -> None:
        self._feedback = feedback

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
                            feedback_args[arg_name] = feedback_input.value
                        else:
                            raise ValueError(
                                f"Argument {arg_name} could not be determined!"
                            )
                    else:
                        raise ValueError(
                            f"Selector {selector} did not match span {span_name} "
                            f"with attributes {span_attributes}"
                        )

                # Create an evaluation span as a child of the current span
                feedback_result = self._feedback.imp(**feedback_args)
                state = self._get_state_arg(func, instance, args, kwargs)
                state.append(feedback_result)

            span_callbacks = kwargs.pop(TRULENS_SPAN_END_CALLBACKS, [])
            span_callbacks.append(span_end_callback)
            kwargs[TRULENS_SPAN_END_CALLBACKS] = span_callbacks

            # Execute the original function first
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def _get_state_arg(func, instance, args, kwargs) -> Any:
        """
        Get the state argument from the function signature. This assumes that
        it's the first non-instance parameter.

        Args:
            func: The function to get the state argument from.
            instance: The instance of the class that the function is
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
