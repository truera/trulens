from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
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

    # If this selector describes a list, which of the following two should we
    # do:
    # 1. [True] Call the feedback function once passing in the entire list.
    # 2. [False] Call the feedback function separately for each entry in the
    #            list and aggregate.
    collect_list: bool = True

    # Whether to only match spans where no ancestor span also matched as well.
    # This is useful for cases where we may have multiple spans match a criteria
    # but we only want to match the first one in a stack trace. For example, in
    # recursive functions, we would only match the first call to the function.
    match_only_if_no_ancestor_matched: bool = False

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
        collect_list: bool = True,
        match_only_if_no_ancestor_matched: bool = False,
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
        self.collect_list = collect_list
        self.match_only_if_no_ancestor_matched = (
            match_only_if_no_ancestor_matched
        )

    def describes_same_spans(self, other: Selector) -> bool:
        return (
            self.function_name == other.function_name
            and self.span_name == other.span_name
            and self.span_type == other.span_type
            and self.match_only_if_no_ancestor_matched
            == other.match_only_if_no_ancestor_matched
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

    def matches_span(
        self, name: Optional[str], attributes: Dict[str, Any]
    ) -> bool:
        ret = True
        if self.function_name is not None:
            ret = ret and self._matches_function_name(
                attributes.get(SpanAttributes.CALL.FUNCTION, None)
            )
        if self.span_name is not None:
            ret = ret and self.span_name == name
        if self.span_type is not None:
            ret = ret and self.span_type == attributes.get(
                SpanAttributes.SPAN_TYPE, None
            )
        return ret

    def process_span(
        self, span_id: str, attributes: Dict[str, Any]
    ) -> FeedbackFunctionInput:
        ret = FeedbackFunctionInput(
            span_id=span_id, collect_list=self.collect_list
        )
        if self.span_attributes_processor is not None:
            ret.value = self.span_attributes_processor(attributes)
        else:
            if self.span_attribute is not None:
                ret.span_attribute = self.span_attribute
            elif self.function_attribute is not None:
                if self.function_attribute == "return":
                    ret.span_attribute = SpanAttributes.CALL.RETURN
                else:
                    ret.span_attribute = f"{SpanAttributes.CALL.KWARGS}.{self.function_attribute}"
            else:
                raise ValueError(
                    "None of `span_attributes_processor`, `span_attribute`, or `function_attribute` are set!"
                )
            ret.value = attributes.get(ret.span_attribute, None)
        return ret

    @staticmethod
    def select_record_input() -> Selector:
        """Returns a `Selector` that gets the record input.

        Returns:
            `Selector` that gets the record input.
        """
        return Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        )

    @staticmethod
    def select_record_output() -> Selector:
        """Returns a `Selector` that gets the record output.

        Returns:
            `Selector` that gets the record output.
        """
        return Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        )

    @staticmethod
    def select_context(*, collect_list: bool) -> Selector:
        """Returns a `Selector` that tries to retrieve contexts.

        Args:
            collect_list:
                Assuming the returned `Selector` describes a list of strings,
                whether to call the feedback function:
                1. [if collect_list is True]:
                        Once giving the entire list as input.
                2. [if collect_list is False]:
                        Separately for each entry in the list and aggregate the
                        results.

        Returns:
            `Selector` that tries to retrieve contexts.
        """

        def context_retrieval_processor(
            attributes: Dict[str, Any],
        ) -> List[str]:
            for curr in [
                SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                SpanAttributes.CALL.RETURN,
            ]:
                ret = attributes.get(curr)
                if (
                    ret is not None
                    and isinstance(ret, list)
                    and all(isinstance(item, str) for item in ret)
                ):
                    return ret
            raise ValueError(
                f"Could not find contexts in attributes: {attributes}"
            )

        return Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attributes_processor=context_retrieval_processor,
            collect_list=collect_list,
            match_only_if_no_ancestor_matched=True,
        )
