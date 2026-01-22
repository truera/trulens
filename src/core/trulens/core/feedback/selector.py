from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.utils.trace_compression import compress_trace_for_feedback
from trulens.core.utils.trace_compression import safe_truncate
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


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

    def _clean_plan_for_cortex(self, plan_str: str) -> str:
        """
        Clean plan content by removing debug messages for Cortex compatibility.

        Args:
            plan_str: Raw plan string that may contain debug messages

        Returns:
            Cleaned plan string with debug messages removed
        """
        import re

        # Patterns to remove - ONLY obvious debug/error messages, be very conservative
        patterns_to_remove = [
            r"^Agent error: [^\n]*\n?",  # Remove lines that START with "Agent error: "
            r"^DEBUG: [^\n]*\n?",  # Remove lines that START with "DEBUG: "
            r"^Query ID: [^\n]*\n?",  # Remove lines that START with "Query ID: "
        ]

        cleaned_plan = plan_str
        for pattern in patterns_to_remove:
            cleaned_plan = re.sub(pattern, "", cleaned_plan, flags=re.MULTILINE)

        # Only remove completely empty lines, preserve all other content
        lines = cleaned_plan.split("\n")
        cleaned_lines = []

        for line in lines:
            if line.strip():  # Keep any line with content
                cleaned_lines.append(line)

        # Join back with single newlines, preserve original formatting
        final_cleaned = "\n".join(cleaned_lines)
        return final_cleaned if final_cleaned.strip() else plan_str

    def to_compressed_json(self, default_handler: Callable = str) -> str:
        """
        Convert trace events to compressed JSON format.
        This reduces token usage while preserving essential information.

        Args:
            default_handler: Function to handle non-serializable objects

        Returns:
            Compressed JSON string representation of the trace
        """
        # First convert to regular JSON
        if self.events is not None:
            trace_data = self.events.to_json(default_handler=default_handler)

            # Try to parse and inspect the structure
            try:
                if isinstance(trace_data, str):
                    parsed = json.loads(trace_data)
                    if isinstance(parsed, dict) and "events" in parsed:
                        pass
            except Exception as e:
                logger.debug(f"Failed to parse trace_data: {e}")
        else:
            # If no events, create minimal trace data
            trace_data = json.dumps({
                "events": [],
                "processed_content_roots": [],
            })

        # Apply compression with explicit plan preservation
        compressed_trace = compress_trace_for_feedback(
            trace_data, preserve_plan=True, target_token_limit=100000
        )

        # Convert compressed data back to JSON string with error handling
        try:
            result = json.dumps(
                compressed_trace, default=default_handler, ensure_ascii=False
            )

            # Check size - Allow up to 100k tokens (roughly 400k characters)
            # Using conservative estimate of 4 characters per token
            MAX_SIZE = 400000  # 400KB limit for 100k token compatibility
            if len(result) > MAX_SIZE:
                logger.debug(
                    f"Compressed trace is too large ({len(result)} chars), reducing to essentials for Cortex compatibility"
                )
                # Keep plan AND tool_execution_evidence for plan adherence evaluation
                essential_trace = {
                    "compressed": True,
                    "size_warning": f"Original trace was {len(result)} characters, reduced for LLM compatibility",
                    "trace_summary": "Large trace compressed to essential elements for Cortex LLM",
                }

                if isinstance(compressed_trace, dict):
                    # Preserve ALL keys containing "plan" (case-insensitive)
                    for key, value in compressed_trace.items():
                        if "plan" in key.lower():
                            essential_trace[key] = value

                    # CRITICAL: Preserve tool_execution_evidence for plan adherence verification
                    if "tool_execution_evidence" in compressed_trace:
                        tool_evidence = compressed_trace[
                            "tool_execution_evidence"
                        ]
                        # Truncate individual items if needed, but keep the structure
                        if isinstance(tool_evidence, dict):
                            truncated_evidence = {}
                            for key, items in tool_evidence.items():
                                if isinstance(items, list):
                                    # Keep first 10 items of each category, truncate content
                                    truncated_items = []
                                    for item in items[:10]:
                                        if isinstance(item, dict):
                                            truncated_item = {}
                                            for k, v in item.items():
                                                if (
                                                    isinstance(v, str)
                                                    and len(v) > 500
                                                ):
                                                    truncated_item[k] = (
                                                        safe_truncate(v, 500)
                                                    )
                                                else:
                                                    truncated_item[k] = v
                                            truncated_items.append(
                                                truncated_item
                                            )
                                        else:
                                            truncated_items.append(item)
                                    truncated_evidence[key] = truncated_items
                                else:
                                    truncated_evidence[key] = items
                            essential_trace["tool_execution_evidence"] = (
                                truncated_evidence
                            )
                        else:
                            essential_trace["tool_execution_evidence"] = (
                                tool_evidence
                            )

                    # Also keep minimal execution flow if no plan keys found
                    if not any(
                        "plan" in k.lower() for k in essential_trace.keys()
                    ):
                        execution_flow = compressed_trace.get(
                            "execution_flow", []
                        )
                        if execution_flow:
                            essential_trace["execution_flow"] = execution_flow[
                                :3
                            ]
                result = json.dumps(
                    essential_trace, default=str, ensure_ascii=False
                )

                # Double-check the reduced size
                if len(result) > MAX_SIZE:
                    # If still too large, keep only keys with "plan" in them
                    all_plan_data = {}
                    if isinstance(compressed_trace, dict):
                        for key, value in compressed_trace.items():
                            if "plan" in key.lower():
                                all_plan_data[key] = value

                    # Use the first plan key found, or combine if multiple
                    plan_data = None
                    if len(all_plan_data) == 1:
                        plan_data = list(all_plan_data.values())[0]
                    elif len(all_plan_data) > 1:
                        # If multiple plan keys, combine them
                        plan_data = all_plan_data
                    else:
                        # Fallback to looking for "plan" key specifically
                        plan_data = (
                            compressed_trace.get("plan")
                            if isinstance(compressed_trace, dict)
                            else None
                        )

                    # If plan itself is too large, clean and truncate it aggressively
                    if (
                        plan_data and len(str(plan_data)) > MAX_SIZE - 500
                    ):  # Leave room for JSON structure
                        plan_str = str(plan_data)

                        # First, apply plan cleaning to remove debug messages
                        cleaned_plan = self._clean_plan_for_cortex(plan_str)

                        # Then truncate if still too large
                        max_plan_size = MAX_SIZE - 500
                        if len(cleaned_plan) > max_plan_size:
                            # Try to find a good truncation point (end of sentence, etc.)
                            truncate_at = (
                                max_plan_size - 100
                            )  # Leave room for truncation message

                            # Look for natural break points
                            for break_point in [". ", "}\n", "],", "\n\n"]:
                                last_break = cleaned_plan.rfind(
                                    break_point, 0, truncate_at
                                )
                                if (
                                    last_break > truncate_at // 2
                                ):  # Don't truncate too early
                                    truncate_at = last_break + len(break_point)
                                    break

                            # Use safe truncation to avoid breaking escape sequences
                            truncated_part = cleaned_plan[:truncate_at]
                            # Remove any trailing backslash
                            while truncated_part.endswith("\\"):
                                truncated_part = truncated_part[:-1]
                            plan_data = (
                                truncated_part
                                + "... [PLAN TRUNCATED - LARGE DATA DETECTED]"
                            )
                        else:
                            plan_data = cleaned_plan

                    minimal_trace = {
                        "compressed": True,
                        "trace_summary": "Trace reduced to plan and evidence due to common LLM context window limits.",
                    }

                    # Add all plan-related data
                    if isinstance(plan_data, dict) and len(all_plan_data) > 1:
                        # If we have multiple plan keys, preserve them all
                        minimal_trace.update(all_plan_data)
                    else:
                        # Single plan or fallback
                        minimal_trace["plan"] = plan_data

                    # CRITICAL: Also preserve minimal tool_execution_evidence
                    if (
                        isinstance(compressed_trace, dict)
                        and "tool_execution_evidence" in compressed_trace
                    ):
                        tool_evidence = compressed_trace[
                            "tool_execution_evidence"
                        ]
                        if isinstance(tool_evidence, dict):
                            # Create a very minimal summary of execution evidence
                            minimal_evidence = {}
                            for category, items in tool_evidence.items():
                                if isinstance(items, list) and items:
                                    # Keep only first 3 items, heavily truncated
                                    minimal_items = []
                                    for item in items[:3]:
                                        if isinstance(item, dict):
                                            mini_item = {
                                                k: (
                                                    safe_truncate(str(v), 200)
                                                    if isinstance(v, str)
                                                    and len(v) > 200
                                                    else v
                                                )
                                                for k, v in item.items()
                                            }
                                            minimal_items.append(mini_item)
                                    if minimal_items:
                                        minimal_evidence[category] = (
                                            minimal_items
                                        )
                            if minimal_evidence:
                                minimal_trace["tool_execution_evidence"] = (
                                    minimal_evidence
                                )
                    result = json.dumps(
                        minimal_trace, default=str, ensure_ascii=False
                    )

                    # Final safety check - if STILL too large, create absolute minimal structure
                    if len(result) > MAX_SIZE:
                        absolute_minimal = {
                            "compressed": True,
                            "plan": "Large plan detected and truncated for Cortex",
                            "trace_summary": "Ultra-minimal trace",
                        }
                        result = json.dumps(
                            absolute_minimal, default=str, ensure_ascii=False
                        )

                        # Ultimate fallback - if even this is too large (shouldn't happen)
                        if len(result) > MAX_SIZE:
                            result = '{"compressed":true,"plan":"truncated","summary":"minimal"}'

            # Validate it's proper JSON by parsing it back
            try:
                json.loads(result)

                # Check for potential encoding issues
                try:
                    result.encode("utf-8")
                except UnicodeEncodeError as e:
                    logger.debug(f"UTF-8 encoding failed: {e}")
                    # Replace problematic characters
                    result = result.encode("utf-8", errors="replace").decode(
                        "utf-8"
                    )

                # Extra safety check - ensure JSON ends properly
                if not result.strip().endswith("}"):
                    # Try to fix by ensuring proper closure
                    if result.strip().endswith(","):
                        result = result.strip()[:-1] + "}"

                # Final validation after all fixes
                try:
                    json.loads(result)
                except json.JSONDecodeError as final_err:
                    logger.debug(
                        f"Final validation failed even after fixes: {final_err}"
                    )
                    result = json.dumps(
                        {
                            "compressed": True,
                            "plan": "JSON validation failed",
                            "trace_summary": "Fallback structure",
                        },
                        default=str,
                        ensure_ascii=False,
                    )

                # Ultimate size check before returning
                if len(result) > MAX_SIZE:
                    logger.debug(
                        f"CRITICAL - Final result still too large ({len(result)} chars), forcing safe truncation"
                    )
                    # Create a guaranteed small, valid JSON structure
                    safe_minimal = {
                        "compressed": True,
                        "plan": "Plan truncated for Cortex compatibility",
                        "trace_summary": "Minimal safe structure",
                    }
                    result = json.dumps(
                        safe_minimal, default=str, ensure_ascii=False
                    )

                # Final JSON validation with repair if needed
                try:
                    json.loads(result)
                except json.JSONDecodeError:
                    # Emergency fallback - guaranteed valid JSON
                    emergency_fallback = '{"compressed":true,"plan":"Emergency fallback","error":"JSON validation failed"}'
                    result = emergency_fallback

                return result
            except json.JSONDecodeError as json_err:
                # If JSON is malformed, create a minimal valid structure
                minimal_fallback = {
                    "compressed": True,
                    "json_validation_error": str(json_err)[:100],
                    "plan": "Plan available but JSON validation failed",
                    "trace_summary": "JSON validation failed, minimal structure provided",
                }
                fallback_result = json.dumps(
                    minimal_fallback, default=str, ensure_ascii=False
                )
                return fallback_result
        except Exception as e:
            logger.warning(f"Error serializing compressed trace: {e}")
            # Fallback to basic trace structure that won't break the LLM
            fallback = {
                "compressed": True,
                "serialization_error": f"Failed to serialize: {str(e)[:100]}",
                "plan": compressed_trace.get("plan")
                if isinstance(compressed_trace, dict)
                else None,
                "trace_summary": "Trace compression encountered serialization issues",
            }
            return json.dumps(fallback, default=str, ensure_ascii=False)


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
    def select_record_input(ignore_none_values: bool = True) -> Selector:
        """Returns a `Selector` that gets the record input.

        Args:
            ignore_none_values: If True, skip evaluation when the input is None.
                Defaults to True to prevent errors on missing data.

        Returns:
            `Selector` that gets the record input.
        """
        return Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            ignore_none_values=ignore_none_values,
        )

    @staticmethod
    def select_record_output(ignore_none_values: bool = True) -> Selector:
        """Returns a `Selector` that gets the record output.

        Args:
            ignore_none_values: If True, skip evaluation when the output is None.
                Defaults to True to prevent errors on missing data.

        Returns:
            `Selector` that gets the record output.
        """
        return Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
            ignore_none_values=ignore_none_values,
        )

    @staticmethod
    def select_context(
        *, collect_list: bool, ignore_none_values: bool = True
    ) -> Selector:
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
            ignore_none_values: If True, skip evaluation when contexts are None.
                Defaults to True to prevent errors on missing data.

        Returns:
            `Selector` that tries to retrieve contexts.
        """

        def context_retrieval_processor(
            attributes: Dict[str, Any],
        ) -> Optional[List[str]]:
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
            # Return None instead of raising if we can't find contexts
            # The ignore_none_values flag will determine whether to skip
            return None

        return Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attributes_processor=context_retrieval_processor,
            collect_list=collect_list,
            match_only_if_no_ancestor_matched=True,
            ignore_none_values=ignore_none_values,
        )
