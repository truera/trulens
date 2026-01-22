"""LlamaIndex Workflow instrumentation.

This implements automatic step tracing by instrumenting the workflow executor
and the step callable classes at class level, similar to how TruGraph
instruments LangGraph's TaskFunction.__call__.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import logging
from typing import Any, Callable, ClassVar, Optional

from opentelemetry.trace import get_current_span
from pydantic import Field
from trulens.core import app as core_app
from trulens.core.otel.instrument import instrument_method
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils
from trulens.otel.semconv.constants import TRULENS_INSTRUMENT_WRAPPER_FLAG
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
        """Ensure class-level instrumentation for workflow components."""
        if cls._is_instrumented:
            return

        cls._is_instrumented = True

        # Set up FunctionAgent instrumentation at class level
        cls._setup_function_agent_instrumentation()

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

        # Also instrument any agents within the workflow
        self._instrument_workflow_agents(app)

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

        # Disable default tool instrumentation before setup to prevent duplicates
        self._disable_default_tool_instrumentation()

        # Set up custom instrumentation that excludes BaseTool methods to avoid duplicates
        kwargs["instrument"] = self._create_custom_instrument()

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

    def _create_custom_instrument(self):
        """Create a custom instrument that excludes BaseTool methods to avoid duplicate spans."""
        try:
            from trulens.apps.llamaindex.tru_llama import LlamaInstrument

            # Import the tool classes we want to exclude
            try:
                from llama_index.tools.types import AsyncBaseTool
                from llama_index.tools.types import BaseTool

                # Get the default methods but filter out BaseTool methods
                default_methods = LlamaInstrument.Default.METHODS
                filtered_methods = []

                for method in default_methods:
                    # Skip BaseTool and AsyncBaseTool methods to avoid duplicates
                    if hasattr(method, "cls") and (
                        method.cls == BaseTool or method.cls == AsyncBaseTool
                    ):
                        continue
                    filtered_methods.append(method)

                # Create custom instrument with filtered methods
                class CustomLlamaInstrument(LlamaInstrument):
                    def __init__(self, *args, **kwargs):
                        # Override the default methods to exclude BaseTool
                        kwargs.setdefault("include_methods", filtered_methods)
                        super().__init__(*args, **kwargs)

                return CustomLlamaInstrument(app=self)

            except ImportError:
                return LlamaInstrument(app=self)

        except ImportError:
            return None

    def _disable_default_tool_instrumentation(self):
        """Disable default BaseTool instrumentation to prevent duplicates."""
        try:
            from llama_index.tools.types import AsyncBaseTool
            from llama_index.tools.types import BaseTool
            from trulens.otel.semconv.constants import (
                TRULENS_INSTRUMENT_WRAPPER_FLAG,
            )

            # Mark BaseTool methods as already instrumented to prevent default instrumentation
            tool_classes = [BaseTool]
            try:
                tool_classes.append(AsyncBaseTool)
            except ImportError:
                pass

            for tool_class in tool_classes:
                for method_name in ["__call__", "call", "acall"]:
                    if hasattr(tool_class, method_name):
                        method = getattr(tool_class, method_name)
                        if not hasattr(method, TRULENS_INSTRUMENT_WRAPPER_FLAG):
                            # Mark as instrumented to prevent default instrumentation
                            setattr(
                                method, TRULENS_INSTRUMENT_WRAPPER_FLAG, True
                            )

        except ImportError as e:
            logger.debug(f"Could not disable default tool instrumentation: {e}")

    @classmethod
    def _update_span_name(cls, span_name: str) -> None:
        """Update current span name if possible."""
        try:
            current_span = get_current_span()
            if current_span and hasattr(current_span, "update_name"):
                current_span.update_name(span_name)
        except Exception as e:
            logger.debug(f"Failed to update span name to {span_name}: {e}")

    @classmethod
    def _setup_function_agent_instrumentation(cls) -> None:
        """Set up class-level instrumentation for FunctionAgent methods."""
        logger.info(
            "TruLlamaWorkflow._setup_function_agent_instrumentation called"
        )
        # Apply instrumentation when TruLlamaWorkflow is initialized
        result = _setup_function_agent_instrumentation_on_demand()
        logger.info(f"FunctionAgent instrumentation result: {result}")

    def _instrument_workflow_agents(self, workflow: Any) -> None:
        """Instrument individual agents within the workflow."""
        logger.info(f"Instrumenting agents in workflow: {type(workflow)}")

        # Debug: Show all attributes of the workflow
        workflow_attrs = [
            attr for attr in dir(workflow) if not attr.startswith("_")
        ]
        logger.info(f"Workflow attributes: {workflow_attrs}")

        # Check if the workflow has agents
        if hasattr(workflow, "agents") and workflow.agents:
            logger.info(
                f"Found {len(workflow.agents)} items in workflow.agents"
            )

            # Debug: Check what's actually in workflow.agents
            for i, agent in enumerate(workflow.agents):
                agent_type = type(agent).__name__
                agent_module = type(agent).__module__
                agent_name = getattr(agent, "name", "unnamed")
                logger.info(
                    f"Agent {i}: {agent_module}.{agent_type} - name: '{agent_name}' - value: {agent}"
                )

            # The agents list contains strings, not agent objects. Let's find the actual agents.

            # Check for agent registry or mapping
            agent_objects = []

            # Search through all attributes for agent objects
            for attr_name in dir(workflow):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr_value = getattr(workflow, attr_name)
                    # Check if it's a FunctionAgent or similar
                    if hasattr(attr_value, "__class__"):
                        class_name = attr_value.__class__.__name__
                        if (
                            "Agent" in class_name
                            or "agent" in class_name.lower()
                        ):
                            agent_objects.append(attr_value)
                        # Also check if it's a dict/list that might contain agents
                        elif isinstance(attr_value, dict):
                            for key, value in attr_value.items():
                                if hasattr(value, "__class__") and (
                                    "Agent" in value.__class__.__name__
                                    or "agent"
                                    in value.__class__.__name__.lower()
                                ):
                                    agent_objects.append(value)
                        elif isinstance(attr_value, (list, tuple)):
                            for i, item in enumerate(attr_value):
                                if hasattr(item, "__class__") and (
                                    "Agent" in item.__class__.__name__
                                    or "agent"
                                    in item.__class__.__name__.lower()
                                ):
                                    agent_objects.append(item)
                except Exception:
                    # Skip attributes that can't be accessed
                    pass

            # Also check private attributes that might contain agents
            for attr_name in [
                "_agent_registry",
                "_agents",
                "_agent_map",
                "_agent_dict",
                "_agent_instances",
            ]:
                if hasattr(workflow, attr_name):
                    try:
                        attr_value = getattr(workflow, attr_name)
                        if isinstance(attr_value, dict):
                            for key, value in attr_value.items():
                                if hasattr(value, "__class__") and (
                                    "Agent" in value.__class__.__name__
                                    or "agent"
                                    in value.__class__.__name__.lower()
                                ):
                                    agent_objects.append(value)
                        elif isinstance(attr_value, (list, tuple)):
                            for i, item in enumerate(attr_value):
                                if hasattr(item, "__class__") and (
                                    "Agent" in item.__class__.__name__
                                    or "agent"
                                    in item.__class__.__name__.lower()
                                ):
                                    agent_objects.append(item)
                    except Exception:
                        # Skip attributes that can't be accessed
                        pass

            # Instrument any agent objects we found
            agent_names = []
            for i, agent in enumerate(agent_objects):
                agent_name = getattr(agent, "name", f"Agent_{i}")
                agent_names.append(agent_name)
                self._instrument_individual_agent(agent)
                # Note: Tool child spans are now created within agent method instrumentation

            if len(agent_names) < 3:
                logger.warning(
                    f"Expected 3 agents (ResearchAgent, WriteAgent, ReviewAgent) but only found {len(agent_names)}"
                )

        else:
            logger.info(
                "No agents found in workflow or workflow.agents is empty"
            )

        # Also check for other possible agent containers
        for attr_name in [
            "_agents",
            "agent_list",
            "nodes",
            "agent_registry",
            "_agent_registry",
        ]:
            if hasattr(workflow, attr_name):
                attr_value = getattr(workflow, attr_name)
                logger.info(
                    f"Found potential agent container: {attr_name} = {attr_value}"
                )

    def _instrument_individual_agent(self, agent: Any) -> None:
        """Instrument an individual agent's methods."""
        agent_name = getattr(agent, "name", "UnknownAgent")
        logger.info(f"Instrumenting agent: {agent_name} ({type(agent)})")

        # Get agent tools for function name extraction
        agent_tools = []
        if hasattr(agent, "tools") and agent.tools:
            agent_tools = agent.tools
            tool_names = [
                getattr(tool, "__name__", str(tool)) for tool in agent_tools
            ]
            logger.info(f"Agent {agent_name} tools: {tool_names}")

        # Get all callable methods of the agent
        methods = [
            method
            for method in dir(agent)
            if not method.startswith("_") and callable(getattr(agent, method))
        ]
        logger.info(f"Agent {agent_name} methods: {methods}")

        # Try to instrument common agent methods - focus on primary execution methods
        # Note: Excluding 'handle_tool_call_results' as it's a post-processing method that creates redundant spans
        # Agent methods are for agent reasoning/orchestration, tool methods are for tool invocations
        agent_methods = [
            "run",
            "arun",
            "chat",
            "achat",
            "stream_chat",
            "astream_chat",
            "__call__",
            "acall",
            "run_agent_step",
            "aggregate_tool_results",
        ]
        # Tool-calling methods should be marked as TOOL spans
        tool_methods = [
            "call_tool",
            "take_step",  # take_step often involves tool execution
        ]
        methods_to_try = agent_methods + tool_methods

        for method_name in methods_to_try:
            if hasattr(agent, method_name):
                method = getattr(agent, method_name)
                if callable(method):
                    # Determine if this is a tool method or agent method
                    is_tool_method = method_name in tool_methods
                    logger.info(
                        f"Instrumenting {agent_name}.{method_name} as {'TOOL' if is_tool_method else 'AGENT'}"
                    )
                    try:
                        # For Pydantic models, we need to use a different approach
                        # Instead of replacing the method, we'll monkey-patch the class
                        original_method = method

                        # Create the instrumented wrapper
                        def create_instrumented_method(
                            original_func,
                            agent_name,
                            method_name,
                            agent_tools,
                            is_tool_method,
                        ):
                            from trulens.core.otel.instrument import instrument

                            def agent_attributes(
                                ret, exception, *args, **kwargs
                            ):
                                """Extract attributes from agent method calls."""

                                # Extract input message/query (skip self)
                                input_msg = None
                                if len(args) > 1:  # First arg is self
                                    input_msg = args[1]
                                elif "message" in kwargs:
                                    input_msg = kwargs["message"]
                                elif "query" in kwargs:
                                    input_msg = kwargs["query"]
                                elif "user_msg" in kwargs:
                                    input_msg = kwargs["user_msg"]

                                # Try to extract function/tool name from the response or context
                                function_name = None

                                # Check if return value has tool call information
                                if hasattr(ret, "sources") and ret.sources:
                                    # Check if any sources contain tool call information
                                    for source in ret.sources:
                                        if hasattr(source, "tool_name"):
                                            function_name = source.tool_name
                                            break
                                        elif hasattr(
                                            source, "metadata"
                                        ) and isinstance(source.metadata, dict):
                                            function_name = source.metadata.get(
                                                "tool_name"
                                            ) or source.metadata.get(
                                                "function_name"
                                            )
                                            if function_name:
                                                break

                                # Check if return value has tool calls directly
                                if (
                                    not function_name
                                    and hasattr(ret, "tool_calls")
                                    and ret.tool_calls
                                ):
                                    if len(ret.tool_calls) > 0:
                                        first_tool_call = ret.tool_calls[0]
                                        if hasattr(
                                            first_tool_call, "tool_name"
                                        ):
                                            function_name = (
                                                first_tool_call.tool_name
                                            )
                                        elif hasattr(
                                            first_tool_call, "function"
                                        ) and hasattr(
                                            first_tool_call.function, "name"
                                        ):
                                            function_name = (
                                                first_tool_call.function.name
                                            )

                                # If we couldn't extract from response, try to infer from agent tools
                                if not function_name and agent_tools:
                                    # Try to get better tool names
                                    def get_tool_name(tool):
                                        # Try multiple ways to get a good tool name
                                        if hasattr(
                                            tool, "metadata"
                                        ) and hasattr(tool.metadata, "name"):
                                            return tool.metadata.name
                                        elif hasattr(tool, "name"):
                                            return tool.name
                                        elif hasattr(tool, "_fn") and hasattr(
                                            tool._fn, "__name__"
                                        ):
                                            return tool._fn.__name__
                                        elif hasattr(tool, "__name__"):
                                            return tool.__name__
                                        else:
                                            return f"tool_{id(tool) % 1000}"  # Use a short ID instead of full object repr

                                    # For single tool agents, we can be confident about the function name
                                    if len(agent_tools) == 1:
                                        function_name = get_tool_name(
                                            agent_tools[0]
                                        )
                                    else:
                                        # Multiple tools - get clean names for all
                                        tool_names = [
                                            get_tool_name(tool)
                                            for tool in agent_tools
                                        ]
                                        function_name = (
                                            f"one_of[{','.join(tool_names)}]"
                                        )

                                # Create span name that includes tool information when available
                                if (
                                    function_name
                                    and not function_name.startswith("one_of[")
                                ):
                                    span_name = f"{agent_name}.{function_name}"
                                else:
                                    span_name = agent_name

                                self._update_span_name(span_name)

                                # Serialize input and output
                                input_ser = _to_json_safe(input_msg)
                                output_ser = _to_json_safe(ret)
                                exception_str = _safe_str(exception)

                                attrs = {
                                    SpanAttributes.WORKFLOW.AGENT_NAME: agent_name,
                                    SpanAttributes.WORKFLOW.INPUT_EVENT: input_ser,
                                    SpanAttributes.WORKFLOW.OUTPUT_EVENT: output_ser,
                                    SpanAttributes.WORKFLOW.ERROR: exception_str,
                                    # Also set generic call attributes for UI panels
                                    SpanAttributes.CALL.RETURN: output_ser,
                                }

                                # Add function name if we found one
                                if function_name:
                                    attrs[
                                        f"{SpanAttributes.CALL.KWARGS}.function_name"
                                    ] = function_name

                                # Add input to call kwargs for UI
                                if input_msg is not None:
                                    attrs[
                                        f"{SpanAttributes.CALL.KWARGS}.message"
                                    ] = input_ser

                                return attrs

                            # Apply the instrument decorator with appropriate span type
                            span_type = (
                                SpanAttributes.SpanType.TOOL
                                if is_tool_method
                                else SpanAttributes.SpanType.AGENT
                            )
                            return instrument(
                                span_type=span_type,
                                attributes=agent_attributes,
                            )(original_func)

                        # Create the instrumented method
                        instrumented_method = create_instrumented_method(
                            original_method,
                            agent_name,
                            method_name,
                            agent_tools,
                            is_tool_method,
                        )

                        # Try to replace the method on the instance
                        if hasattr(agent, "__dict__"):
                            # For regular objects
                            agent.__dict__[method_name] = (
                                instrumented_method.__get__(agent, type(agent))
                            )
                        else:
                            # For Pydantic models, try to set the attribute directly
                            object.__setattr__(
                                agent,
                                method_name,
                                instrumented_method.__get__(agent, type(agent)),
                            )

                        logger.debug(
                            f"Successfully wrapped {agent_name}.{method_name}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Instance-level instrumentation failed for {agent_name}.{method_name}: {e}"
                        )

                        # Try alternative approach: monkey-patch the class method
                        try:
                            agent_class = type(agent)
                            if hasattr(agent_class, method_name):
                                # Create attributes function for class-level instrumentation
                                def create_class_attributes_func():
                                    def class_agent_attributes(
                                        ret, exception, *args, **kwargs
                                    ):
                                        # Extract agent name from self (first argument)
                                        current_agent_name = (
                                            getattr(
                                                args[0], "name", "UnknownAgent"
                                            )
                                            if args
                                            else "UnknownAgent"
                                        )

                                        # Extract input
                                        input_msg = None
                                        if len(args) > 1:
                                            input_msg = args[1]
                                        else:
                                            input_msg = kwargs.get(
                                                "message",
                                                kwargs.get(
                                                    "query",
                                                    kwargs.get("user_msg"),
                                                ),
                                            )

                                        # Try to extract function name from agent tools
                                        function_name = None
                                        if (
                                            args
                                            and hasattr(args[0], "tools")
                                            and args[0].tools
                                        ):
                                            current_agent_tools = args[0].tools

                                            # Try to get better tool names
                                            def get_tool_name(tool):
                                                if hasattr(
                                                    tool, "metadata"
                                                ) and hasattr(
                                                    tool.metadata, "name"
                                                ):
                                                    return tool.metadata.name
                                                elif hasattr(tool, "name"):
                                                    return tool.name
                                                elif hasattr(
                                                    tool, "_fn"
                                                ) and hasattr(
                                                    tool._fn, "__name__"
                                                ):
                                                    return tool._fn.__name__
                                                elif hasattr(tool, "__name__"):
                                                    return tool.__name__
                                                else:
                                                    return f"tool_{id(tool) % 1000}"

                                            if len(current_agent_tools) == 1:
                                                function_name = get_tool_name(
                                                    current_agent_tools[0]
                                                )
                                            else:
                                                # Multiple tools - get clean names
                                                tool_names = [
                                                    get_tool_name(tool)
                                                    for tool in current_agent_tools
                                                ]
                                                function_name = f"one_of[{','.join(tool_names)}]"

                                        # Create span name that includes tool information when available
                                        if (
                                            function_name
                                            and not function_name.startswith(
                                                "one_of["
                                            )
                                        ):
                                            span_name = f"{current_agent_name}.{function_name}"
                                        else:
                                            span_name = current_agent_name

                                        TruLlamaWorkflow._update_span_name(
                                            span_name
                                        )

                                        attrs = {
                                            SpanAttributes.WORKFLOW.AGENT_NAME: current_agent_name,
                                            SpanAttributes.WORKFLOW.INPUT_EVENT: _to_json_safe(
                                                input_msg
                                            ),
                                            SpanAttributes.WORKFLOW.OUTPUT_EVENT: _to_json_safe(
                                                ret
                                            ),
                                            SpanAttributes.WORKFLOW.ERROR: _safe_str(
                                                exception
                                            ),
                                            SpanAttributes.CALL.RETURN: _to_json_safe(
                                                ret
                                            ),
                                        }

                                        # Add function name if we found one
                                        if function_name:
                                            attrs[
                                                f"{SpanAttributes.CALL.KWARGS}.function_name"
                                            ] = function_name

                                        return attrs

                                    return class_agent_attributes

                                # Apply instrumentation at class level with appropriate span type
                                class_span_type = (
                                    SpanAttributes.SpanType.TOOL
                                    if is_tool_method
                                    else SpanAttributes.SpanType.AGENT
                                )
                                instrument_method(
                                    cls=agent_class,
                                    method_name=method_name,
                                    span_type=class_span_type,
                                    attributes=create_class_attributes_func(),
                                )
                        except Exception as e2:
                            logger.debug(
                                f"Class-level instrumentation also failed for {method_name}: {e2}"
                            )

    def _instrument_agent_tools(self, agent: Any) -> None:
        """Instrument individual tools within an agent for better span naming."""
        agent_name = getattr(agent, "name", "UnknownAgent")

        if not hasattr(agent, "tools") or not agent.tools:
            return

        for i, tool in enumerate(agent.tools):
            tool_name = getattr(tool, "__name__", f"tool_{i}")

            # Try to get a better name for the tool
            if hasattr(tool, "metadata") and hasattr(tool.metadata, "name"):
                tool_name = tool.metadata.name
            elif hasattr(tool, "name"):
                tool_name = tool.name
            elif hasattr(tool, "_fn") and hasattr(tool._fn, "__name__"):
                tool_name = tool._fn.__name__
            elif hasattr(tool, "__name__"):
                tool_name = tool.__name__

            # Instrument the tool's call methods
            methods_to_instrument = ["__call__", "call", "acall"]

            for method_name in methods_to_instrument:
                if hasattr(tool, method_name):
                    try:
                        original_method = getattr(tool, method_name)
                        if callable(original_method):
                            # Create instrumented wrapper for tool
                            def create_tool_wrapper(
                                orig_method, t_name, m_name, a_name
                            ):
                                def tool_attributes(
                                    ret, exception, *args, **kwargs
                                ):
                                    """Extract attributes from tool calls."""

                                    # Extract input (skip self)
                                    input_data = None
                                    if len(args) > 1:
                                        input_data = args[1]
                                    elif len(args) == 1 and not hasattr(
                                        args[0], "__class__"
                                    ):
                                        # If first arg is not self (i.e., not an object)
                                        input_data = args[0]
                                    elif kwargs:
                                        input_data = kwargs

                                    # Use just the tool name since this will be a child span under the agent
                                    span_name = t_name
                                    self._update_span_name(span_name)

                                    return {
                                        SpanAttributes.WORKFLOW.AGENT_NAME: a_name,
                                        SpanAttributes.WORKFLOW.INPUT_EVENT: _to_json_safe(
                                            input_data
                                        ),
                                        SpanAttributes.WORKFLOW.OUTPUT_EVENT: _to_json_safe(
                                            ret
                                        ),
                                        SpanAttributes.WORKFLOW.ERROR: _safe_str(
                                            exception
                                        ),
                                        SpanAttributes.CALL.RETURN: _to_json_safe(
                                            ret
                                        ),
                                        f"{SpanAttributes.CALL.KWARGS}.function_name": t_name,
                                    }

                                # Create a wrapper that ensures proper parent-child relationship
                                @functools.wraps(orig_method)
                                def tool_wrapper(*args, **kwargs):
                                    from opentelemetry import trace
                                    from trulens.experimental.otel_tracing.core.session import (
                                        TRULENS_SERVICE_NAME,
                                    )

                                    # Get the current span (should be the agent span)
                                    current_span = trace.get_current_span()
                                    if (
                                        current_span
                                        and current_span.is_recording()
                                    ):
                                        # Create a child span explicitly
                                        tracer = trace.get_tracer_provider().get_tracer(
                                            TRULENS_SERVICE_NAME
                                        )
                                        with tracer.start_as_current_span(
                                            t_name
                                        ) as tool_span:
                                            # Set tool span attributes
                                            tool_span.set_attribute(
                                                SpanAttributes.SPAN_TYPE,
                                                SpanAttributes.SpanType.TOOL,
                                            )
                                            tool_span.set_attribute(
                                                SpanAttributes.WORKFLOW.AGENT_NAME,
                                                a_name,
                                            )
                                            tool_span.set_attribute(
                                                f"{SpanAttributes.CALL.KWARGS}.function_name",
                                                t_name,
                                            )

                                            try:
                                                result = orig_method(
                                                    *args, **kwargs
                                                )
                                                tool_span.set_attribute(
                                                    SpanAttributes.CALL.RETURN,
                                                    _to_json_safe(result),
                                                )
                                                return result
                                            except Exception as e:
                                                tool_span.set_attribute(
                                                    SpanAttributes.WORKFLOW.ERROR,
                                                    _safe_str(e),
                                                )
                                                raise
                                    else:
                                        # Fallback to original method if no active span
                                        return orig_method(*args, **kwargs)

                                return tool_wrapper

                            # Create and apply the wrapper
                            instrumented_method = create_tool_wrapper(
                                original_method,
                                tool_name,
                                method_name,
                                agent_name,
                            )

                            # Replace the method
                            setattr(tool, method_name, instrumented_method)

                    except Exception as e:
                        # Log at debug level since these failures are often expected
                        logger.debug(
                            f"Could not instrument {tool_name}.{method_name}: {e}"
                        )

    def _create_agent_method_wrapper(
        self, agent: Any, method_name: str, original_method: Callable
    ) -> Callable:
        """Create a wrapper for an agent method that adds tracing."""
        from trulens.core.otel.instrument import instrument

        agent_name = getattr(agent, "name", "UnknownAgent")

        def agent_attributes(ret, exception, *args, **kwargs):
            """Extract attributes from agent method calls."""
            # Extract input message/query
            input_msg = None
            if len(args) > 0:  # First arg after self
                input_msg = args[0]
            elif "message" in kwargs:
                input_msg = kwargs["message"]
            elif "query" in kwargs:
                input_msg = kwargs["query"]

            # Update span name to include agent name
            span_name = f"{agent_name}.{method_name}"
            self._update_span_name(span_name)

            # Serialize input and output
            input_ser = _to_json_safe(input_msg)
            output_ser = _to_json_safe(ret)
            exception_str = _safe_str(exception)

            attrs = {
                SpanAttributes.WORKFLOW.AGENT_NAME: agent_name,
                SpanAttributes.WORKFLOW.INPUT_EVENT: input_ser,
                SpanAttributes.WORKFLOW.OUTPUT_EVENT: output_ser,
                SpanAttributes.WORKFLOW.ERROR: exception_str,
                # Also set generic call attributes for UI panels
                SpanAttributes.CALL.RETURN: output_ser,
            }

            # Add input to call kwargs for UI
            if input_msg is not None:
                attrs[f"{SpanAttributes.CALL.KWARGS}.message"] = input_ser

            return attrs

        # Apply the instrument decorator
        instrumented_method = instrument(
            span_type=SpanAttributes.SpanType.AGENT,
            attributes=agent_attributes,
        )(original_method)

        return instrumented_method


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


# Apply FunctionAgent instrumentation when TruLlamaWorkflow is used
# This ensures FunctionAgent methods are instrumented when the workflow is actually used
def _setup_function_agent_instrumentation_on_demand():
    """Set up FunctionAgent instrumentation when needed."""
    if not is_otel_tracing_enabled():
        logger.debug("OTEL not enabled, skipping FunctionAgent instrumentation")
        return

    try:
        # Try to import FunctionAgent
        from llama_index.core.agent.workflow import FunctionAgent

        logger.info(f"Successfully imported FunctionAgent: {FunctionAgent}")

        # Debug: Check what methods FunctionAgent actually has
        all_methods = [
            method
            for method in dir(FunctionAgent)
            if not method.startswith("_")
        ]
        logger.info(f"FunctionAgent available methods: {all_methods}")

    except ImportError as e:
        logger.debug(
            f"FunctionAgent not available, skipping instrumentation: {e}"
        )
        return

    # Check if FunctionAgent methods are already instrumented
    # Let's start with the most likely methods that exist
    methods_to_instrument = [
        "run",
        "arun",
        "chat",
        "achat",
        "stream_chat",
        "astream_chat",
        "__call__",
        "acall",
    ]

    instrumented_any = False
    for method_name in methods_to_instrument:
        if hasattr(FunctionAgent, method_name):
            method = getattr(FunctionAgent, method_name)
            if hasattr(method, TRULENS_INSTRUMENT_WRAPPER_FLAG):
                logger.debug(
                    f"FunctionAgent.{method_name} already instrumented"
                )
                continue

            logger.info(
                f"Applying instrumentation to FunctionAgent.{method_name}"
            )

            # Create attributes function for FunctionAgent calls
            def create_function_agent_attributes(method_name):
                def function_agent_attributes(ret, exception, *args, **kwargs):
                    """Extract attributes from FunctionAgent method calls."""
                    # Get the agent instance (self)
                    agent_instance = None
                    if args and hasattr(args[0], "name"):
                        agent_instance = args[0]

                    # Extract input message/query
                    input_msg = None
                    if len(args) > 1:
                        input_msg = args[1]
                    elif "message" in kwargs:
                        input_msg = kwargs["message"]
                    elif "query" in kwargs:
                        input_msg = kwargs["query"]

                    # Extract agent name and update span name
                    agent_name = None
                    if agent_instance and hasattr(agent_instance, "name"):
                        agent_name = agent_instance.name
                        # Update span name to include agent name
                        span_name = f"{agent_name}.{method_name}"
                        TruLlamaWorkflow._update_span_name(span_name)

                    # Serialize input and output
                    input_ser = _to_json_safe(input_msg)
                    output_ser = _to_json_safe(ret)
                    exception_str = _safe_str(exception)

                    attrs = {
                        SpanAttributes.WORKFLOW.AGENT_NAME: agent_name,
                        SpanAttributes.WORKFLOW.INPUT_EVENT: input_ser,
                        SpanAttributes.WORKFLOW.OUTPUT_EVENT: output_ser,
                        SpanAttributes.WORKFLOW.ERROR: exception_str,
                        # Also set generic call attributes for UI panels
                        SpanAttributes.CALL.RETURN: output_ser,
                    }

                    # Add input to call kwargs for UI
                    if input_msg is not None:
                        attrs[f"{SpanAttributes.CALL.KWARGS}.message"] = (
                            input_ser
                        )

                    return attrs

                return function_agent_attributes

            try:
                instrument_method(
                    cls=FunctionAgent,
                    method_name=method_name,
                    span_type=SpanAttributes.SpanType.AGENT,
                    attributes=create_function_agent_attributes(method_name),
                )
                instrumented_any = True
            except Exception as e:
                logger.debug(
                    f"Could not instrument FunctionAgent.{method_name}: {e}"
                )

    if instrumented_any:
        logger.info("Successfully instrumented FunctionAgent methods")

    return instrumented_any
