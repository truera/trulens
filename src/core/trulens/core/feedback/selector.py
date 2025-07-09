from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.otel.semconv.trace import SpanAttributes


@dataclass
class ProcessedContentNode:
    event: pd.Series
    content: Any
    parent: Optional[ProcessedContentNode]
    children: List[ProcessedContentNode]


@dataclass
class Trace:
    # Filtered events in the trace that match the selector. That is, Each row
    # corresponds to a `trulens.core.schema.event.Event` along with an
    # additional column `processed_content` that contains the processed content
    # of the event described by the following fields of the selector:
    # 1. span_attributes_processor
    # 2. span_attribute
    # 3. function_attribute
    # Use this over `processed_content_roots` if the hierarchy of the events
    # (i.e. parent-child relationship) is not important.
    events: Optional[pd.DataFrame]

    # This describes the DAG of the filtered events in the trace.
    # Example:
    # If we have three events pre-filtering A, B, and C where:
    # 1. A is the parent of B and E.
    # 2. B is the parent of C.
    # 3. C is the parent of D.
    # 4. B, E, and D are kept and A and C are filtered out.
    # Then `processed_content_roots` will contain two events B and E where B has
    # a single child D.
    # Pre-filtering:
    #       A
    #      / \
    #     B   E
    #     |
    #     C
    #     |
    #     D
    # Post-filtering:
    #     B    E
    #     |
    #     D
    # Use this over `events` if the hierarchy of the events (i.e. parent-child
    # relationship) is important.
    processed_content_roots: List[ProcessedContentNode]

    def __init__(self):
        self.events = None
        self.processed_content_roots = []

    def add_event(
        self,
        processed_content: Any,
        event: pd.Series,
        parent_processed_content_node: Optional[ProcessedContentNode],
    ) -> ProcessedContentNode:
        event_with_processed_content = event.copy()
        event_with_processed_content["processed_content"] = processed_content
        if self.events is None:
            self.events = pd.DataFrame(event_with_processed_content).transpose()
        else:
            self.events = pd.concat(
                [
                    self.events,
                    pd.DataFrame(event_with_processed_content).transpose(),
                ],
                ignore_index=True,
            )
        node = ProcessedContentNode(
            event=event,
            content=processed_content,
            parent=parent_processed_content_node,
            children=[],
        )
        if parent_processed_content_node is not None:
            parent_processed_content_node.children.append(node)
        else:
            self.processed_content_roots.append(node)
        return node


@dataclass
class Selector:
    # The following fields describe how to select the spans.
    # If `trace_level` is True, then this `Selector` doesn't describe a single
    # span but any number of spans in the trace. It will describe all spans
    # unless further filtered by `function_name`, `span_name`, or `span_type`.
    # If `trace_level` is False, then this `Selector` describes a single span
    # determined by `function_name`, `span_name`, and `span_type`.
    trace_level: bool = False
    function_name: Optional[str] = None
    span_name: Optional[str] = None
    span_type: Optional[str] = None

    # The following fields describe what to extract from the spans to use as
    # the feedback function input.
    span_attributes_processor: Optional[Callable[[Dict[str, Any]], Any]] = None
    span_attribute: Optional[str] = None
    function_attribute: Optional[str] = None

    # If the value extracted from the span is None (i.e. from
    # `span_attributes_processor`, `span_attribute`, or `function_attribute`),
    # then whether to ignore it or not.
    ignore_none_values: bool = False

    # If this selector describes a list, which of the following two should we
    # do:
    # 1. [True] Call the feedback function once passing in the entire list.
    # 2. [False] Call the feedback function separately for each entry in the
    #            list and aggregate.
    # Ignored when `trace_level` is True.
    collect_list: bool = True

    # Whether to only match spans where no ancestor span also matched as well.
    # This is useful for cases where we may have multiple spans match a criteria
    # but we only want to match the first one in a stack trace. For example, in
    # recursive functions, we would only match the first call to the function.
    # Ignored when `trace_level` is True.
    match_only_if_no_ancestor_matched: bool = False

    def __init__(
        self,
        trace_level: bool = False,
        function_name: Optional[str] = None,
        span_name: Optional[str] = None,
        span_type: Optional[str] = None,
        span_attributes_processor: Optional[
            Callable[[Dict[str, Any]], Any]
        ] = None,
        span_attribute: Optional[str] = None,
        function_attribute: Optional[str] = None,
        ignore_none_values: bool = False,
        collect_list: bool = True,
        match_only_if_no_ancestor_matched: bool = False,
    ):
        if (
            not trace_level
            and sum([
                span_attributes_processor is not None,
                span_attribute is not None,
                function_attribute is not None,
            ])
            != 1
        ):
            raise ValueError(
                "Must specify exactly one of `span_attributes_processor`, `span_attribute`, or `function_attribute`!"
            )
        self.trace_level = trace_level
        self.function_name = function_name
        self.span_name = span_name
        self.span_type = span_type
        self.span_attributes_processor = span_attributes_processor
        self.span_attribute = span_attribute
        self.function_attribute = function_attribute
        self.ignore_none_values = ignore_none_values
        self.collect_list = collect_list
        self.match_only_if_no_ancestor_matched = (
            match_only_if_no_ancestor_matched
        )

    def describes_same_spans(self, other: Selector) -> bool:
        return (
            self.trace_level == other.trace_level
            and self.function_name == other.function_name
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
            elif not self.trace_level:
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
