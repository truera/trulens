from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from trulens.otel.semconv.trace import SpanAttributes


@dataclass
class Selector:
    # The following fields describe how to select the spans.
    function_name: Optional[str] = None
    span_name: Optional[str] = None
    span_type: Optional[str] = None

    # The following fields describe what to extract from the spans to use as
    # the feedback function input.
    span_attributes_processor: Optional[Callable[[Dict[str, Any]], Any]] = None
    span_attribute: Optional[str] = None
    function_attribute: Optional[str] = None

    def __init__(
        self,
        function_name: Optional[str] = None,
        span_name: Optional[str] = None,
        span_type: Optional[str] = None,
        span_attributes_processor: Optional[
            Callable[[Dict[str, Any]], Any]
        ] = None,
        span_attribute: Optional[str] = None,
        function_attribute: Optional[str] = None,
    ):
        if function_name is None and span_name is None and span_type is None:
            raise ValueError(
                "Must specify at least one of `function_name`, `span_name`, or `span_type`!"
            )
        if (
            sum([
                span_attributes_processor is not None,
                span_attribute is not None,
                function_attribute is not None,
            ])
            != 1
        ):
            raise ValueError(
                "Must specify exactly one of `span_attributes_processor`, `span_attribute`, or `function_attribute`!"
            )
        self.function_name = function_name
        self.span_name = span_name
        self.span_type = span_type
        self.span_attributes_processor = span_attributes_processor
        self.span_attribute = span_attribute
        self.function_attribute = function_attribute

    def describes_same_spans(self, other: Selector) -> bool:
        return (
            self.function_name == other.function_name
            and self.span_name == other.span_name
            and self.span_type == other.span_type
        )

    @staticmethod
    def _split_function_name(function_name: str) -> List[str]:
        if "::" in function_name:
            return function_name.split("::")
        return function_name.split(".")

    def _matches_function_name(self, function_name: Optional[str]) -> bool:
        if self.function_name is None:
            return True
        if function_name is None:
            return False
        actual = self._split_function_name(function_name)
        expected = self._split_function_name(self.function_name)
        if len(actual) < len(expected):
            return False
        return actual[-len(expected) :] == expected

    def matches_span(self, attributes: Dict[str, Any]) -> bool:
        ret = True
        if self.function_name is not None:
            ret = ret and self._matches_function_name(
                attributes.get(SpanAttributes.CALL.FUNCTION, None)
            )
        if self.span_name is not None:
            ret = ret and self.span_name == attributes.get("name", None)
        if self.span_type is not None:
            ret = ret and self.span_type == attributes.get(
                SpanAttributes.SPAN_TYPE, None
            )
        return ret

    def process_span(self, attributes: Dict[str, Any]) -> Any:
        if self.span_attributes_processor is not None:
            return self.span_attributes_processor(attributes)
        if self.span_attribute is not None:
            return attributes.get(self.span_attribute, None)
        if self.function_attribute is not None:
            if self.function_attribute == "return":
                return attributes.get(SpanAttributes.CALL.RETURN, None)
            return attributes.get(
                f"{SpanAttributes.CALL.KWARGS}.{self.function_attribute}", None
            )
        raise ValueError(
            "None of `span_attributes_processor`, `span_attribute`, or `function_attribute` are set!"
        )
