"""LangGraph app instrumentation."""

import dataclasses
from functools import wraps
import inspect
from inspect import BoundArguments
from inspect import Signature
import json
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    List,
    Optional,
)

from opentelemetry.trace import get_current_span
from pydantic import Field
from trulens.apps.langchain.tru_chain import TruChain
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.instruments import InstrumentedMethod
from trulens.core.otel.function_call_context_manager import (
    create_function_call_context_manager,
)
from trulens.core.otel.instrument import instrument_function
from trulens.core.otel.instrument import instrument_method
from trulens.core.otel.instrument import set_general_span_attributes
from trulens.core.otel.instrument import set_user_defined_attributes
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils
from trulens.otel.semconv.constants import TRULENS_INSTRUMENT_WRAPPER_FLAG
from trulens.otel.semconv.trace import SpanAttributes

from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command

logger = logging.getLogger(__name__)

# Handle backward compatibility for TaskFunction rename
try:
    from langgraph.func import TaskFunction
except ImportError:
    try:
        from langgraph.func import _TaskFunction as TaskFunction
    except ImportError:
        logger.warning("TaskFunction not found, skipping instrumentation")
        TaskFunction = None


class LangGraphInstrument(core_instruments.Instrument):
    """Instrumentation for LangGraph apps."""

    class Default:
        """Instrumentation specification for LangGraph apps."""

        MODULES = {"langgraph"}
        """Modules by prefix to instrument."""

        CLASSES = lambda: {
            Pregel,
            StateGraph,
            Command,
            # Note: TaskFunction (or _TaskFunction) is instrumented at class-level during initialization
        }
        """Classes to instrument."""

        METHODS: List[InstrumentedMethod] = []

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LangGraphInstrument.Default.MODULES,
            include_classes=LangGraphInstrument.Default.CLASSES(),
            include_methods=LangGraphInstrument.Default.METHODS,
            *args,
            **kwargs,
        )


class TruGraph(TruChain):
    """Recorder for _LangGraph_ applications.

    This recorder is designed for LangGraph apps, providing a way to instrument,
    log, and evaluate their behavior while inheriting all LangChain instrumentation
    capabilities.

    **Automatic LangGraph Function Tracing**:

    TruGraph automatically instruments LangGraph components including:
    - `Pregel` methods (invoke, ainvoke, stream, etc.) - instrumented at class-level during import
    - `TaskFunction.__call__` method (or `_TaskFunction.__call__` in newer versions) - instrumented at class-level during import
    - `StateGraph` objects (uncompiled graphs) for logging/debugging purposes

    **Class-Level Instrumentation**: Both `@task` functions (TaskFunction/_TaskFunction) and
    `Pregel` graph methods are instrumented at the class level when TruGraph is imported.
    This ensures all function calls are captured regardless of where the instances
    are embedded in the object hierarchy (e.g., inside custom classes).

    **Benefits of Class-Level Approach**:
    - **Guaranteed Coverage**: All TaskFunction/_TaskFunction and Pregel method calls are captured
    - **No Import Timing Issues**: Works regardless of when objects are created
    - **Consistent Span Types**: Properly sets "graph_node" and "graph_task" span types

    Example: "Creating a LangGraph multi-agent application"

        Consider an example LangGraph multi-agent application:

        ```python
        from langgraph.graph import StateGraph, MessagesState
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from langchain_community.tools.tavily_search import TavilySearchResults

        # Create agents
        llm = ChatOpenAI(model="gpt-4")
        search_tool = TavilySearchResults()
        research_agent = create_react_agent(llm, [search_tool])

        # Build graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("researcher", research_agent)
        workflow.add_edge("researcher", END)
        workflow.set_entry_point("researcher")

        graph = workflow.compile()
        ```

    Example: "Using @task and @entrypoint decorators"

        ```python
        from langgraph.func import task, entrypoint

        @task
        def my_task_function(state):
            ...
            return updated_state

        @entrypoint()
        def my_workflow(input_state):
            result = my_task_function(input_state).result()
            return result
        ```

    Example: "Custom class with multiple LangGraph workflows"

        ```python
        class ComplexAgent:
            def __init__(self):
                self.planner = StateGraph(...).compile()
                self.executor = StateGraph(...).compile()
                self.critic = StateGraph(...).compile()

            def run(self, query):
                plan = self.planner.invoke({"input": query})
                execution = self.executor.invoke({"plan": plan})
                critique = self.critic.invoke({"execution": execution})
                return self.synthesize_results(plan, execution, critique)

        # Both of these work:
        tru_graph = TruGraph(graph)  # Direct LangGraph
        tru_agent = TruGraph(ComplexAgent())  # Custom class
        ```

    The application can be wrapped in a `TruGraph` recorder to provide logging
    and evaluation upon the application's use.

    Example: "Using the `TruGraph` recorder"

        ```python
        from trulens.apps.langgraph import TruGraph

        # Wrap application
        tru_recorder = TruGraph(
            graph,
            app_name="MultiAgentApp",
            app_version="v1",
            feedbacks=[f_context_relevance]
        )

        # Record application runs
        with tru_recorder as recording:
            result = graph.invoke({"messages": [("user", "What is the weather?")]})
        ```

    Args:
        app: A LangGraph application. Can be:
            - `Pregel`: A compiled LangGraph
            - `StateGraph`: An uncompiled LangGraph (will be auto-compiled)
            - Custom class: Any object that uses LangGraph workflows internally

        **kwargs: Additional arguments to pass to [App][trulens.core.app.App]
            and [AppDefinition][trulens.core.schema.app.AppDefinition].
    """

    # Class-level flag to track instrumentation status
    _is_instrumented: ClassVar[bool] = False

    app: Any
    """The application to be instrumented. Can be LangGraph objects or custom classes."""

    # Fix the root_callable field to have the correct default
    root_callable: ClassVar[Optional[pyschema_utils.FunctionOrMethod]] = Field(
        default=None
    )
    """The root callable of the wrapped app."""

    @classmethod
    def _setup_task_function_instrumentation(cls):
        """Set up class-level instrumentation for TaskFunction/__call__ (or _TaskFunction.__call__)"""

        if TaskFunction is None:
            logger.debug(
                "TaskFunction/_TaskFunction not available, skipping instrumentation"
            )
            return

        if not is_otel_tracing_enabled():
            logger.debug(
                "OTEL not enabled, skipping TaskFunction/_TaskFunction class-level instrumentation"
            )
            return

        try:
            # Check if TaskFunction.__call__ is already instrumented
            if hasattr(TaskFunction.__call__, TRULENS_INSTRUMENT_WRAPPER_FLAG):
                logger.debug(
                    "TaskFunction/_TaskFunction.__call__ already instrumented"
                )
                return

            logger.info(
                f"Applying class-level instrumentation to {TaskFunction.__name__}.__call__"
            )

            # Create attributes function for TaskFunction calls
            def task_function_attributes(ret, exception, *args, **kwargs):
                attributes = {}

                # For TaskFunction.__call__, the first argument is self (TaskFunction/_TaskFunction instance)
                if args and len(args) > 0:
                    task_function_instance = args[0]
                    task_args = args[1:] if len(args) > 1 else ()
                    task_kwargs = kwargs

                    if hasattr(task_function_instance, "func") and hasattr(
                        task_function_instance.func, "__name__"
                    ):
                        task_name = task_function_instance.func.__name__
                        attributes[SpanAttributes.GRAPH_TASK.TASK_NAME] = (
                            task_name
                        )

                        # Update the span name to the task name
                        try:
                            current_span = get_current_span()
                            if current_span and hasattr(
                                current_span, "update_name"
                            ):
                                current_span.update_name(task_name)
                        except Exception as e:
                            logger.debug(
                                f"Failed to update span name to {task_name}: {e}"
                            )

                    # Serialize the task input arguments
                    try:
                        if hasattr(task_function_instance, "func"):
                            try:
                                sig = inspect.signature(
                                    task_function_instance.func
                                )

                                # Filter kwargs to only include those that match the signature
                                sig_params = list(sig.parameters.keys())
                                filtered_kwargs = {
                                    k: v
                                    for k, v in task_kwargs.items()
                                    if k in sig_params
                                }

                                args_as_kwargs = {}
                                for i, arg in enumerate(task_args):
                                    if i < len(sig_params):
                                        args_as_kwargs[sig_params[i]] = arg

                                # Merge positional args (as kwargs) with existing kwargs
                                all_kwargs = {
                                    **args_as_kwargs,
                                    **filtered_kwargs,
                                }

                                # Try to bind with all arguments as kwargs
                                if all_kwargs:
                                    bound_args = sig.bind_partial(**all_kwargs)
                                    bound_args.apply_defaults()

                                    # Collect all arguments into a single JSON structure
                                    input_args = {}
                                    for (
                                        name,
                                        value,
                                    ) in bound_args.arguments.items():
                                        try:
                                            if hasattr(
                                                value, "__dict__"
                                            ) and hasattr(
                                                value, "__dataclass_fields__"
                                            ):
                                                input_args[name] = (
                                                    dataclasses.asdict(value)
                                                )
                                            elif hasattr(value, "model_dump"):
                                                input_args[name] = (
                                                    value.model_dump()
                                                )
                                            elif hasattr(
                                                value, "dict"
                                            ) and callable(
                                                getattr(value, "dict")
                                            ):
                                                input_args[name] = value.dict()
                                            else:
                                                input_args[name] = value
                                        except Exception:
                                            input_args[name] = str(value)

                                    attributes[
                                        SpanAttributes.GRAPH_TASK.INPUT_STATE
                                    ] = json.dumps(
                                        input_args, default=str, indent=2
                                    )
                                else:
                                    attributes[
                                        SpanAttributes.GRAPH_TASK.INPUT_STATE
                                    ] = "{}"

                            except Exception as bind_error:
                                logger.debug(
                                    f"Argument binding failed: {bind_error}, using fallback"
                                )

                                # Collect arguments as simple structure
                                fallback_args = {}
                                for i, arg in enumerate(task_args):
                                    try:
                                        fallback_args[f"arg_{i}"] = arg
                                    except Exception:
                                        fallback_args[f"arg_{i}"] = str(arg)

                                for name, value in task_kwargs.items():
                                    try:
                                        fallback_args[name] = value
                                    except Exception:
                                        fallback_args[name] = str(value)

                                attributes[
                                    SpanAttributes.GRAPH_TASK.INPUT_STATE
                                ] = json.dumps(
                                    fallback_args, default=str, indent=2
                                )
                        else:
                            attributes[
                                SpanAttributes.GRAPH_TASK.INPUT_STATE
                            ] = json.dumps(
                                {"args": task_args, "kwargs": task_kwargs},
                                default=str,
                                indent=2,
                            )

                    except Exception as e:
                        logger.warning(
                            f"Error processing task input arguments: {e}"
                        )
                        attributes[SpanAttributes.GRAPH_TASK.INPUT_STATE] = (
                            f"Error processing args: {str(e)}"
                        )

                # Handle return value (Future object for now)
                if ret is not None and not exception:
                    try:
                        # For now, capture the Future object info
                        # TODO: In future, we might want to capture the actual result when task Future completes
                        attributes[SpanAttributes.GRAPH_TASK.OUTPUT_STATE] = (
                            str(ret)
                        )
                    except Exception:
                        attributes[SpanAttributes.GRAPH_TASK.OUTPUT_STATE] = (
                            str(ret)
                        )

                if exception:
                    attributes[SpanAttributes.GRAPH_TASK.ERROR] = str(exception)

                return attributes

            # Apply the instrumentation at class level
            instrument_method(
                cls=TaskFunction,
                method_name="__call__",
                span_type=SpanAttributes.SpanType.GRAPH_TASK,
                attributes=task_function_attributes,
            )

            logger.info(
                f"Successfully applied class-level instrumentation to {TaskFunction.__name__}.__call__"
            )

        except Exception as e:
            logger.warning(
                f"Failed to apply class-level TaskFunction/_TaskFunction instrumentation: {e}"
            )

    @classmethod
    def _setup_internal_node_instrumentation(cls):
        """Set up instrumentation for internal LangGraph node execution"""

        if not is_otel_tracing_enabled():
            logger.debug(
                "OTEL not enabled, skipping internal node instrumentation"
            )
            return

        try:
            # Try to instrument internal LangGraph execution methods
            # These are the methods that actually call individual node functions

            # Look for common internal execution patterns in LangGraph
            internal_methods = [
                # Common patterns from LangGraph internals that handle node execution
                "_execute_node",
                "_invoke_node",
                "_run_node",
                "_call_node",
                "invoke_node",
                "run_node",
                "_execute_step",
                "_invoke_step",
            ]

            # Try to find and instrument any of these methods on Pregel
            for method_name in internal_methods:
                if hasattr(Pregel, method_name):
                    try:

                        def create_node_execution_attributes(method_name):
                            def node_execution_attributes(
                                ret, exception, *args, **kwargs
                            ):
                                attributes = {}

                                # Try to extract node name from arguments
                                # LangGraph internal methods often pass node name as first or second arg
                                node_name = "unknown_node"
                                if args:
                                    # Check if first arg looks like a node name (string)
                                    if len(args) > 0 and isinstance(
                                        args[0], str
                                    ):
                                        node_name = args[0]
                                    elif len(args) > 1 and isinstance(
                                        args[1], str
                                    ):
                                        node_name = args[1]
                                    # Also check for node name in common parameter positions
                                    for i, arg in enumerate(
                                        args[:3]
                                    ):  # Check first 3 args
                                        if (
                                            isinstance(arg, str)
                                            and len(arg) < 50
                                        ):  # Reasonable node name length
                                            # Check if it looks like a node name (no spaces, reasonable length)
                                            if " " not in arg and len(arg) > 0:
                                                node_name = arg
                                                break

                                attributes[
                                    SpanAttributes.GRAPH_NODE.NODE_NAME
                                ] = node_name

                                # Try to capture state from arguments
                                if args:
                                    for i, arg in enumerate(args):
                                        if (
                                            isinstance(arg, dict)
                                            and "messages" in arg
                                        ):
                                            messages = arg["messages"]
                                            if messages and hasattr(
                                                messages[-1], "content"
                                            ):
                                                latest_content = messages[
                                                    -1
                                                ].content
                                                attributes[
                                                    SpanAttributes.GRAPH_NODE.INPUT_STATE
                                                ] = latest_content
                                                attributes[
                                                    SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                                ] = latest_content
                                                break

                                # Capture output
                                if ret is not None and not exception:
                                    if (
                                        isinstance(ret, dict)
                                        and "messages" in ret
                                    ):
                                        messages = ret["messages"]
                                        if messages and hasattr(
                                            messages[-1], "content"
                                        ):
                                            latest_content = messages[
                                                -1
                                            ].content
                                            attributes[
                                                SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                            ] = latest_content
                                            attributes[
                                                SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                            ] = latest_content
                                    else:
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                        ] = (
                                            str(ret)[:200] + "..."
                                            if len(str(ret)) > 200
                                            else str(ret)
                                        )

                                if exception:
                                    attributes[
                                        SpanAttributes.GRAPH_NODE.ERROR
                                    ] = str(exception)

                                return attributes

                            return node_execution_attributes

                        instrument_method(
                            cls=Pregel,
                            method_name=method_name,
                            span_type=SpanAttributes.SpanType.GRAPH_NODE,
                            attributes=create_node_execution_attributes(
                                method_name
                            ),
                        )
                        logger.debug(
                            f"Instrumented internal method: Pregel.{method_name}"
                        )

                    except Exception as e:
                        logger.debug(
                            f"Failed to instrument Pregel.{method_name}: {e}"
                        )

        except Exception as e:
            logger.warning(
                f"Failed to set up internal node instrumentation: {e}"
            )

    @classmethod
    def _wrap_stream_generator(cls, original_generator):
        """Wrap a LangGraph stream generator to capture individual node updates as spans"""

        try:

            def instrumented_generator():
                for chunk in original_generator:
                    # Each chunk typically contains node updates
                    if isinstance(chunk, dict):
                        for node_name, node_data in chunk.items():
                            # Create a span for each node update using the proper TruLens approach
                            span_name = f"graph_node.{node_name}"

                            try:
                                with create_function_call_context_manager(
                                    create_new_span=True, span_name=span_name
                                ) as span:
                                    # Set general span attributes
                                    set_general_span_attributes(
                                        span, SpanAttributes.SpanType.GRAPH_NODE
                                    )

                                    # Build attributes dict
                                    attributes = {
                                        SpanAttributes.GRAPH_NODE.NODE_NAME: node_name,
                                    }

                                    # Capture the latest message
                                    if (
                                        isinstance(node_data, dict)
                                        and "messages" in node_data
                                    ):
                                        messages = node_data["messages"]
                                        if messages and hasattr(
                                            messages[-1], "content"
                                        ):
                                            latest_content = messages[
                                                -1
                                            ].content
                                            attributes[
                                                SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                            ] = latest_content
                                            attributes[
                                                SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                            ] = latest_content
                                        else:
                                            attributes[
                                                SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                            ] = str(node_data)
                                    else:
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                        ] = str(node_data)

                                    # Set the user-defined attributes
                                    set_user_defined_attributes(
                                        span, attributes=attributes
                                    )

                            except Exception as e:
                                logger.debug(
                                    f"Failed to create span for node {node_name}: {e}"
                                )

                    # Yield the original chunk unchanged
                    yield chunk

            return instrumented_generator()

        except Exception as e:
            logger.warning(f"Failed to wrap stream generator: {e}")
            return original_generator

    @classmethod
    def _instrument_streaming_method(cls, method_name):
        """Instrument a specific streaming method to capture node-by-node updates"""

        try:
            if not hasattr(Pregel, method_name):
                logger.debug(f"Method {method_name} not found on Pregel")
                return

            original_method = getattr(Pregel, method_name)

            if hasattr(original_method, TRULENS_INSTRUMENT_WRAPPER_FLAG):
                logger.debug(f"Method {method_name} already instrumented")
                return

            def create_instrumented_streaming_method(
                original_method, method_name
            ):
                def instrumented_method(self, *args, **kwargs):
                    # Call the original method to get the generator
                    original_generator = original_method(self, *args, **kwargs)

                    # Wrap the generator to capture individual node updates
                    return cls._wrap_stream_generator(original_generator)

                # Mark as instrumented
                setattr(
                    instrumented_method, TRULENS_INSTRUMENT_WRAPPER_FLAG, True
                )
                return instrumented_method

            # Replace the method with our instrumented version
            instrumented_method = create_instrumented_streaming_method(
                original_method, method_name
            )
            setattr(Pregel, method_name, instrumented_method)

            logger.debug(
                f"Successfully instrumented streaming method: {method_name}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to instrument streaming method {method_name}: {e}"
            )

    @classmethod
    def _setup_pregel_instrumentation(cls):
        """Set up class-level instrumentation for Pregel methods"""

        if not is_otel_tracing_enabled():
            logger.error("OTEL not enabled, skipping Pregel instrumentation")
            return

        try:
            if hasattr(Pregel, "invoke") and hasattr(
                getattr(Pregel, "invoke"), TRULENS_INSTRUMENT_WRAPPER_FLAG
            ):
                logger.debug("Pregel methods already instrumented")
                return

            logger.info(
                "Applying class-level instrumentation to Pregel methods"
            )

            # First, instrument internal node execution methods
            cls._setup_internal_node_instrumentation()

            # Create enhanced attributes function for Pregel methods
            def pregel_attributes(ret, exception, *args, **kwargs):
                attributes = {}

                # Extract method name to understand what we're instrumenting
                frame = inspect.currentframe()
                method_name = None
                try:
                    if frame and frame.f_back and frame.f_back.f_code:
                        method_name = frame.f_back.f_code.co_name
                finally:
                    del frame

                if ret is not None and not exception:
                    # Check if we can extract execution information from the result
                    if isinstance(ret, dict):
                        # Look for patterns that indicate multiple node executions
                        # This is a heuristic approach to detect sequential node processing

                        # Check if this looks like a multi-step result
                        result_keys = list(ret.keys())

                        if len(result_keys) > 1:
                            # Multiple keys might indicate multiple node outputs
                            node_names = []
                            for key in result_keys:
                                if key not in [
                                    "topic",  # Input parameter
                                    "messages",  # State data
                                ]:  # Skip common input/state keys
                                    node_names.append(key)

                            if node_names:
                                attributes[
                                    SpanAttributes.GRAPH_NODE.NODES_EXECUTED
                                ] = json.dumps(node_names)
                                attributes["execution_summary"] = (
                                    f"Executed nodes: {', '.join(node_names)}"
                                )

                                # Add individual node details as attributes (simpler approach)
                                for i, node_name in enumerate(node_names):
                                    attributes[f"detected_node_{i}_name"] = (
                                        node_name
                                    )
                                    if node_name in ret:
                                        node_output = str(ret[node_name])
                                        if len(node_output) > 200:
                                            node_output = (
                                                node_output[:200] + "..."
                                            )
                                        attributes[
                                            f"detected_node_{i}_output"
                                        ] = node_output
                                        attributes[
                                            f"detected_node_{i}_type"
                                        ] = "step_result"

                                        # CREATE INDIVIDUAL SPAN RIGHT HERE!
                                        try:
                                            with create_function_call_context_manager(
                                                create_new_span=True,
                                                span_name=node_name,
                                            ) as span:
                                                # Set general span attributes
                                                set_general_span_attributes(
                                                    span,
                                                    SpanAttributes.SpanType.GRAPH_NODE,
                                                )

                                                # Build attributes for this specific node
                                                node_attributes = {
                                                    SpanAttributes.GRAPH_NODE.NODE_NAME: node_name,
                                                    SpanAttributes.GRAPH_NODE.OUTPUT_STATE: node_output,
                                                    SpanAttributes.GRAPH_NODE.LATEST_MESSAGE: node_output,
                                                }

                                                # Set the user-defined attributes
                                                set_user_defined_attributes(
                                                    span,
                                                    attributes=node_attributes,
                                                )

                                        except Exception as span_e:
                                            logger.exception(
                                                f"Failed to create span for {node_name}: {span_e}"
                                            )

                # Capture input state
                if args and len(args) > 1:
                    input_data = args[1]
                    try:
                        if isinstance(input_data, dict):
                            # Extract latest message for better readability
                            if "messages" in input_data and isinstance(
                                input_data["messages"], list
                            ):
                                messages = input_data["messages"]
                                if messages:
                                    latest_message = messages[-1]
                                    if hasattr(latest_message, "content"):
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.INPUT_STATE
                                        ] = latest_message.content
                                    else:
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.INPUT_STATE
                                        ] = str(latest_message)
                                else:
                                    attributes[
                                        SpanAttributes.GRAPH_NODE.INPUT_STATE
                                    ] = "Empty messages"
                            else:
                                # For non-message states, serialize first level only
                                simplified_input = {}
                                for k, v in input_data.items():
                                    if isinstance(v, (str, int, float, bool)):
                                        simplified_input[k] = v
                                    elif isinstance(v, list) and len(v) <= 3:
                                        simplified_input[k] = v
                                    else:
                                        simplified_input[k] = (
                                            f"<{type(v).__name__}: {len(v) if hasattr(v, '__len__') else 'unknown'}>"
                                        )
                                attributes[
                                    SpanAttributes.GRAPH_NODE.INPUT_STATE
                                ] = json.dumps(
                                    simplified_input, default=str, indent=2
                                )
                        else:
                            attributes[
                                SpanAttributes.GRAPH_NODE.INPUT_STATE
                            ] = str(input_data)
                    except Exception as e:
                        attributes[SpanAttributes.GRAPH_NODE.INPUT_STATE] = (
                            f"Error serializing input: {str(e)}"
                        )

                # Handle kwargs
                for k, v in kwargs.items():
                    if k in ["input", "state", "data"]:
                        try:
                            if isinstance(v, dict) and "messages" in v:
                                messages = v["messages"]
                                if messages and hasattr(
                                    messages[-1], "content"
                                ):
                                    attributes[
                                        SpanAttributes.GRAPH_NODE.INPUT_STATE
                                    ] = messages[-1].content
                                else:
                                    attributes[
                                        SpanAttributes.GRAPH_NODE.INPUT_STATE
                                    ] = str(v)
                            else:
                                attributes[
                                    SpanAttributes.GRAPH_NODE.INPUT_STATE
                                ] = str(v)
                        except Exception:
                            attributes[
                                SpanAttributes.GRAPH_NODE.INPUT_STATE
                            ] = str(v)
                        break

                # Capture output state
                if ret is not None and not exception:
                    try:
                        # For streaming methods, we'll get generators - capture that info
                        if method_name in [
                            "stream",
                            "astream",
                            "stream_mode",
                            "astream_mode",
                        ]:
                            attributes[
                                SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                            ] = f"<Generator: {type(ret).__name__}>"
                        elif isinstance(ret, dict):
                            # Extract latest message for better readability
                            if "messages" in ret and isinstance(
                                ret["messages"], list
                            ):
                                messages = ret["messages"]
                                if messages:
                                    latest_message = messages[-1]
                                    if hasattr(latest_message, "content"):
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                        ] = latest_message.content
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                        ] = latest_message.content
                                    else:
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                        ] = str(latest_message)
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                        ] = str(latest_message)
                                else:
                                    attributes[
                                        SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                    ] = "Empty messages"
                            else:
                                # For non-message states, serialize first level only
                                simplified_output = {}
                                for k, v in ret.items():
                                    if isinstance(v, (str, int, float, bool)):
                                        simplified_output[k] = v
                                    elif isinstance(v, list) and len(v) <= 3:
                                        simplified_output[k] = v
                                    else:
                                        simplified_output[k] = (
                                            f"<{type(v).__name__}: {len(v) if hasattr(v, '__len__') else 'unknown'}>"
                                        )
                                attributes[
                                    SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                ] = json.dumps(
                                    simplified_output, default=str, indent=2
                                )
                        else:
                            attributes[
                                SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                            ] = str(ret)
                    except Exception as e:
                        attributes[SpanAttributes.GRAPH_NODE.OUTPUT_STATE] = (
                            f"Error serializing output: {str(e)}"
                        )

                if exception:
                    attributes[SpanAttributes.GRAPH_NODE.ERROR] = str(exception)

                return attributes

            # Create specialized attributes function for streaming methods
            def streaming_pregel_attributes(ret, exception, *args, **kwargs):
                attributes = {}

                # For streaming methods, we capture the generator info and wrap it
                if ret is not None and not exception:
                    attributes[SpanAttributes.GRAPH_NODE.OUTPUT_STATE] = (
                        f"<Stream Generator: {type(ret).__name__}>"
                    )

                    # Wrap the generator to capture individual node updates as separate spans
                    if hasattr(ret, "__iter__"):
                        attributes["stream_info"] = (
                            "Streaming generator wrapped - individual node updates will be captured as spans"
                        )

                        # Note: We can't modify ret here, so we'll instrument differently

                # Capture input like normal methods
                if args and len(args) > 1:
                    input_data = args[1]
                    try:
                        if isinstance(input_data, dict):
                            if "messages" in input_data and isinstance(
                                input_data["messages"], list
                            ):
                                messages = input_data["messages"]
                                if messages:
                                    latest_message = messages[-1]
                                    if hasattr(latest_message, "content"):
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.INPUT_STATE
                                        ] = latest_message.content
                                    else:
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.INPUT_STATE
                                        ] = str(latest_message)
                            else:
                                simplified_input = {}
                                for k, v in input_data.items():
                                    if isinstance(v, (str, int, float, bool)):
                                        simplified_input[k] = v
                                    elif isinstance(v, list) and len(v) <= 3:
                                        simplified_input[k] = v
                                    else:
                                        simplified_input[k] = (
                                            f"<{type(v).__name__}: {len(v) if hasattr(v, '__len__') else 'unknown'}>"
                                        )
                                attributes[
                                    SpanAttributes.GRAPH_NODE.INPUT_STATE
                                ] = json.dumps(
                                    simplified_input, default=str, indent=2
                                )
                        else:
                            attributes[
                                SpanAttributes.GRAPH_NODE.INPUT_STATE
                            ] = str(input_data)
                    except Exception as e:
                        attributes[SpanAttributes.GRAPH_NODE.INPUT_STATE] = (
                            f"Error serializing input: {str(e)}"
                        )

                if exception:
                    attributes[SpanAttributes.GRAPH_NODE.ERROR] = str(exception)

                return attributes

            # Apply instrumentation to different method groups
            sync_methods = ["invoke"]
            async_methods = ["ainvoke"]
            streaming_methods = [
                "stream",
                "astream",
                "stream_mode",
                "astream_mode",
            ]

            # Regular methods use standard attributes
            for method_name in sync_methods + async_methods:
                instrument_method(
                    cls=Pregel,
                    method_name=method_name,
                    span_type=SpanAttributes.SpanType.GRAPH_NODE,
                    attributes=pregel_attributes,
                )
                logger.debug(
                    f"Applied class-level instrumentation to Pregel.{method_name}"
                )

            # Streaming methods use specialized wrapper approach
            for method_name in streaming_methods:
                cls._instrument_streaming_method(method_name)
                logger.debug(
                    f"Applied streaming instrumentation to Pregel.{method_name}"
                )

            # Node detection will be handled by the enhanced pregel_attributes function
            logger.debug(
                "Successfully applied class-level instrumentation to Pregel methods"
            )

        except Exception as e:
            logger.warning(
                f"Failed to apply class-level Pregel instrumentation: {e}"
            )

    @classmethod
    def _setup_node_function_instrumentation(cls):
        """Set up instrumentation for individual node functions to capture node names"""

        if not is_otel_tracing_enabled():
            logger.debug(
                "OTEL not enabled, skipping node function instrumentation"
            )
            return

        try:
            # We'll add a helper method to instrument individual node functions
            # This will be used when we process the graph structure
            logger.debug("Node function instrumentation setup complete")

        except Exception as e:
            logger.warning(
                f"Failed to apply node function instrumentation: {e}"
            )

    @staticmethod
    def instrument_node_function(
        node_func: Callable, node_name: str
    ) -> Callable:
        """Instrument a single node function to add node name tracking.

        Args:
            node_func: The node function to instrument
            node_name: The name of the node

        Returns:
            The instrumented function
        """
        if not is_otel_tracing_enabled():
            return node_func

        try:

            def node_attributes(ret, exception, *args, **kwargs):
                attributes = {
                    SpanAttributes.GRAPH_NODE.NODE_NAME: node_name,
                }

                # Capture input state from arguments
                if args:
                    state_arg = args[0] if args else {}
                    try:
                        if isinstance(state_arg, dict):
                            # Extract latest message for better readability
                            if "messages" in state_arg and isinstance(
                                state_arg["messages"], list
                            ):
                                messages = state_arg["messages"]
                                if messages:
                                    latest_message = messages[-1]
                                    if hasattr(latest_message, "content"):
                                        message_content = latest_message.content
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.INPUT_STATE
                                        ] = message_content
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                        ] = message_content
                                    else:
                                        message_str = str(latest_message)
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.INPUT_STATE
                                        ] = message_str
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                        ] = message_str
                                else:
                                    attributes[
                                        SpanAttributes.GRAPH_NODE.INPUT_STATE
                                    ] = "Empty messages"
                                    attributes[
                                        SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                    ] = "Empty messages"
                            else:
                                # For non-message states, serialize first level only
                                simplified_input = {}
                                for k, v in state_arg.items():
                                    if isinstance(v, (str, int, float, bool)):
                                        simplified_input[k] = v
                                    elif isinstance(v, list) and len(v) <= 3:
                                        simplified_input[k] = v
                                    else:
                                        simplified_input[k] = (
                                            f"<{type(v).__name__}: {len(v) if hasattr(v, '__len__') else 'unknown'}>"
                                        )
                                attributes[
                                    SpanAttributes.GRAPH_NODE.INPUT_STATE
                                ] = json.dumps(
                                    simplified_input, default=str, indent=2
                                )
                        else:
                            attributes[
                                SpanAttributes.GRAPH_NODE.INPUT_STATE
                            ] = str(state_arg)
                    except Exception as e:
                        attributes[SpanAttributes.GRAPH_NODE.INPUT_STATE] = (
                            f"Error serializing input: {str(e)}"
                        )

                # Capture output state
                if ret is not None and not exception:
                    try:
                        if isinstance(ret, dict):
                            # Extract latest message for better readability
                            if "messages" in ret and isinstance(
                                ret["messages"], list
                            ):
                                messages = ret["messages"]
                                if messages:
                                    latest_message = messages[-1]
                                    if hasattr(latest_message, "content"):
                                        message_content = latest_message.content
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                        ] = message_content
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                        ] = message_content
                                    else:
                                        message_str = str(latest_message)
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                        ] = message_str
                                        attributes[
                                            SpanAttributes.GRAPH_NODE.LATEST_MESSAGE
                                        ] = message_str
                                else:
                                    attributes[
                                        SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                    ] = "Empty messages"
                            else:
                                # For non-message states, serialize first level only
                                simplified_output = {}
                                for k, v in ret.items():
                                    if isinstance(v, (str, int, float, bool)):
                                        simplified_output[k] = v
                                    elif isinstance(v, list) and len(v) <= 3:
                                        simplified_output[k] = v
                                    else:
                                        simplified_output[k] = (
                                            f"<{type(v).__name__}: {len(v) if hasattr(v, '__len__') else 'unknown'}>"
                                        )
                                attributes[
                                    SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                                ] = json.dumps(
                                    simplified_output, default=str, indent=2
                                )
                        else:
                            attributes[
                                SpanAttributes.GRAPH_NODE.OUTPUT_STATE
                            ] = str(ret)
                    except Exception as e:
                        attributes[SpanAttributes.GRAPH_NODE.OUTPUT_STATE] = (
                            f"Error serializing output: {str(e)}"
                        )

                if exception:
                    attributes[SpanAttributes.GRAPH_NODE.ERROR] = str(exception)

                return attributes

            @wraps(node_func)
            def instrumented_node_func(*args, **kwargs):
                # Set span name to the node name
                from opentelemetry.trace import get_current_span

                try:
                    current_span = get_current_span()
                    if current_span and hasattr(current_span, "update_name"):
                        current_span.update_name(node_name)
                except Exception as e:
                    logger.debug(
                        f"Failed to update span name to {node_name}: {e}"
                    )

                # Call the original function
                return node_func(*args, **kwargs)

            # Apply instrumentation
            return instrument_function(
                func=instrumented_node_func,
                span_type=SpanAttributes.SpanType.GRAPH_NODE,
                attributes=node_attributes,
                span_name=node_name,
            )

        except Exception as e:
            logger.warning(
                f"Failed to instrument node function {node_name}: {e}"
            )
            return node_func

    @classmethod
    def _ensure_instrumentation(cls):
        """Ensure one-time initialization of instrumentation."""
        if not cls._is_instrumented:
            cls._setup_task_function_instrumentation()
            cls._setup_pregel_instrumentation()
            cls._setup_node_function_instrumentation()
            cls._is_instrumented = True
        else:
            logger.debug("Instrumentation already set up")

    def __init__(
        self,
        app: Any,
        main_method: Optional[Callable] = None,
        **kwargs: Any,
    ):
        # Ensure instrumentation is set up before initializing
        self._ensure_instrumentation()

        # Only do minimal preparation to avoid interfering with existing instrumentation
        app = self._prepare_app(app)

        kwargs["app"] = app

        connector = kwargs.get("connector")
        if connector is not None:
            TruSession(connector=connector)
        else:
            TruSession()

        if main_method is not None:
            kwargs["main_method"] = main_method
        kwargs["root_class"] = pyschema_utils.Class.of_object(app)
        from trulens.apps.langchain.tru_chain import LangChainInstrument

        class CombinedInstrument(core_instruments.Instrument):
            def __init__(self, *args, **kwargs):
                langchain_default = LangChainInstrument.Default
                langgraph_default = LangGraphInstrument.Default

                combined_modules = langchain_default.MODULES.union(
                    langgraph_default.MODULES
                )
                combined_classes = langchain_default.CLASSES().union(
                    langgraph_default.CLASSES()
                )
                combined_methods = (
                    langchain_default.METHODS + langgraph_default.METHODS
                )

                super().__init__(
                    include_modules=combined_modules,
                    include_classes=combined_classes,
                    include_methods=combined_methods,
                    *args,
                    **kwargs,
                )

        kwargs["instrument"] = CombinedInstrument(app=self)

        core_app.App.__init__(self, **kwargs)

    def _prepare_app(self, app: Any) -> Any:
        """
        Prepare the app for instrumentation by handling different input types.

        Args:
            app: The input application

        Returns:
            The prepared application ready for instrumentation
        """
        if isinstance(app, StateGraph):
            logger.warning(
                "Received uncompiled StateGraph. Compiling it for instrumentation. "
                "For better control, consider compiling the graph yourself before wrapping with TruGraph."
            )
            compiled_app = app.compile()
            self._instrument_graph_nodes(compiled_app)
            return compiled_app  # type: ignore

        if isinstance(app, Pregel):
            self._instrument_graph_nodes(app)
            return app

        return app

    def _instrument_graph_nodes(self, compiled_graph: Any) -> None:
        """
        Instrument individual node functions in a compiled graph for better node name tracking.

        Args:
            compiled_graph: A compiled LangGraph (Pregel instance)
        """
        if not is_otel_tracing_enabled():
            return

        try:
            instrumented_nodes = 0

            # Method 1: Try to access and instrument the direct nodes dictionary
            if hasattr(compiled_graph, "nodes") and isinstance(
                compiled_graph.nodes, dict
            ):
                for node_name, node_obj in compiled_graph.nodes.items():
                    # Handle different node types
                    if hasattr(node_obj, "func") and callable(node_obj.func):
                        # Node with a function attribute
                        original_func = node_obj.func
                        instrumented_func = self.instrument_node_function(
                            original_func, node_name
                        )
                        node_obj.func = instrumented_func
                        logger.debug(
                            f"Instrumented node function via .func: {node_name}"
                        )
                        instrumented_nodes += 1
                    elif callable(node_obj):
                        # Node is directly a callable
                        instrumented_func = self.instrument_node_function(
                            node_obj, node_name
                        )
                        compiled_graph.nodes[node_name] = instrumented_func
                        logger.debug(f"Instrumented callable node: {node_name}")
                        instrumented_nodes += 1

            # Method 2: Try alternative node access patterns
            # Look for other possible node storage patterns in LangGraph
            alternative_node_attrs = ["_nodes", "graph", "_graph", "spec"]
            for attr_name in alternative_node_attrs:
                if hasattr(compiled_graph, attr_name):
                    attr_obj = getattr(compiled_graph, attr_name)
                    if hasattr(attr_obj, "nodes") and isinstance(
                        attr_obj.nodes, dict
                    ):
                        for node_name, node_obj in attr_obj.nodes.items():
                            if callable(node_obj):
                                instrumented_func = (
                                    self.instrument_node_function(
                                        node_obj, node_name
                                    )
                                )
                                attr_obj.nodes[node_name] = instrumented_func
                                logger.debug(
                                    f"Instrumented node via {attr_name}: {node_name}"
                                )
                                instrumented_nodes += 1

            # Method 3: Try to access step definitions which might contain the actual node functions
            if hasattr(compiled_graph, "step") and hasattr(
                compiled_graph.step, "__self__"
            ):
                step_obj = compiled_graph.step.__self__
                if hasattr(step_obj, "nodes") and isinstance(
                    step_obj.nodes, dict
                ):
                    for node_name, node_obj in step_obj.nodes.items():
                        if callable(node_obj):
                            instrumented_func = self.instrument_node_function(
                                node_obj, node_name
                            )
                            step_obj.nodes[node_name] = instrumented_func
                            logger.debug(
                                f"Instrumented node via step: {node_name}"
                            )
                            instrumented_nodes += 1

            if instrumented_nodes > 0:
                logger.info(
                    f"Successfully instrumented {instrumented_nodes} node functions"
                )
            else:
                logger.debug(
                    "Could not find node functions to instrument in compiled graph"
                )
                # Log the structure for debugging
                logger.debug(f"Compiled graph type: {type(compiled_graph)}")
                logger.debug(
                    f"Compiled graph attributes: {[attr for attr in dir(compiled_graph) if not attr.startswith('_')]}"
                )

        except Exception as e:
            logger.warning(f"Failed to instrument graph nodes: {e}")

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """
        if "input" in bindings.arguments:
            temp = bindings.arguments["input"]
            if isinstance(temp, dict):
                if "messages" in temp:
                    messages = temp["messages"]
                    if isinstance(messages, list) and len(messages) > 0:
                        last_message = messages[-1]
                        if hasattr(last_message, "content"):
                            return last_message.content
                        elif (
                            isinstance(last_message, tuple)
                            and len(last_message) > 1
                        ):
                            return last_message[1]  # (role, content) tuple
                        else:
                            return str(last_message)
                elif "query" in temp:
                    return temp["query"]
                else:
                    for _, value in temp.items():
                        if isinstance(value, str):
                            return value
                    return str(temp)
            elif isinstance(temp, str):
                return temp
            else:
                return str(temp)

        return super().main_input(func, sig, bindings)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> str:
        """
        Determine the main output string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """
        if isinstance(ret, dict):
            if "messages" in ret:
                messages = ret["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    last_message = messages[-1]
                    if hasattr(last_message, "content"):
                        return last_message.content
                    elif (
                        isinstance(last_message, tuple)
                        and len(last_message) > 1
                    ):
                        return last_message[1]  # (role, content) tuple
                    else:
                        return str(last_message)
                else:
                    return str(ret)
            else:
                for _, value in ret.items():
                    if isinstance(value, str):
                        return value
                return str(ret)
        elif isinstance(ret, str):
            return ret
        else:
            return super().main_output(func, sig, bindings, ret)

    def main_call(self, human: str):
        """A single text to a single text invocation of this app."""
        if isinstance(self.app, Pregel):
            try:
                result = self.app.invoke({"messages": [("user", human)]})
                return self._extract_output_from_result(result)
            except Exception as e:
                raise Exception(f"Error invoking Langgraph workflow: {str(e)}")
        else:
            raise Exception(
                f"App must be an instance of Pregel, got {type(self.app)}"
            )

    async def main_acall(self, human: str):
        """A single text to a single text async invocation of this app."""
        if isinstance(self.app, Pregel):
            try:
                result = await self.app.ainvoke({"messages": [("user", human)]})
                return self._extract_output_from_result(result)
            except Exception as e:
                raise Exception(f"Error invoking Langgraph workflow: {str(e)}")
        else:
            raise Exception(
                f"App must be an instance of Pregel, got {type(self.app)}"
            )

    def _extract_output_from_result(self, result: Any) -> str:
        """Helper method to extract string output from LangGraph result."""
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                elif isinstance(last_message, tuple) and len(last_message) > 1:
                    return last_message[1]  # (role, content) tuple
                else:
                    return str(last_message)
            else:
                return str(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)


TruGraph.model_rebuild()
