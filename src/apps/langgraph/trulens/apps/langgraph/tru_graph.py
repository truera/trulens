"""LangGraph app instrumentation."""

import dataclasses
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
from trulens.core.otel.instrument import instrument
from trulens.core.otel.instrument import instrument_method
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils
from trulens.experimental.otel_tracing.core.span import (
    set_general_span_attributes,
)
from trulens.experimental.otel_tracing.core.span import (
    set_user_defined_attributes,
)
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

    @staticmethod
    def _extract_latest_message_content(data: dict) -> Optional[str]:
        """Extract latest message content from state dict."""
        if "messages" in data and isinstance(data["messages"], list):
            messages = data["messages"]
            if messages:
                latest_message = messages[-1]
                if hasattr(latest_message, "content"):
                    return latest_message.content
                else:
                    return str(latest_message)
        return None

    @staticmethod
    def _serialize_state_for_attributes(data: dict, max_items: int = 3) -> str:
        """Serialize state dictionary for span attributes."""
        simplified = {}
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool)):
                simplified[k] = v
            elif isinstance(v, list) and len(v) <= max_items:
                simplified[k] = v
            else:
                simplified[k] = (
                    f"<{type(v).__name__}: {len(v) if hasattr(v, '__len__') else 'unknown'}>"
                )
        return json.dumps(simplified, default=str, indent=2)

    @classmethod
    def _build_state_attributes(cls, data: dict, is_input: bool = True) -> dict:
        """Build standard state attributes for spans."""
        attributes = {}

        # Try to extract latest message
        latest_content = cls._extract_latest_message_content(data)
        if latest_content:
            state_key = (
                SpanAttributes.GRAPH_NODE.INPUT_STATE
                if is_input
                else SpanAttributes.GRAPH_NODE.OUTPUT_STATE
            )
            attributes[state_key] = latest_content
            attributes[SpanAttributes.GRAPH_NODE.LATEST_MESSAGE] = (
                latest_content
            )
        else:
            # Fallback to serialized state
            state_key = (
                SpanAttributes.GRAPH_NODE.INPUT_STATE
                if is_input
                else SpanAttributes.GRAPH_NODE.OUTPUT_STATE
            )
            attributes[state_key] = cls._serialize_state_for_attributes(data)

        return attributes

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
    def _handle_input_output_state(
        cls, input_data: Any, ret: Any, exception: Exception, attributes: dict
    ) -> None:
        """Handle input and output state processing for spans."""
        # Capture input state
        if input_data is not None:
            try:
                if isinstance(input_data, dict):
                    input_attrs = cls._build_state_attributes(
                        input_data, is_input=True
                    )
                    attributes.update(input_attrs)
                else:
                    attributes[SpanAttributes.GRAPH_NODE.INPUT_STATE] = str(
                        input_data
                    )
            except Exception as e:
                attributes[SpanAttributes.GRAPH_NODE.INPUT_STATE] = (
                    f"Error serializing input: {str(e)}"
                )

        # Capture output state
        if ret is not None and not exception:
            try:
                if isinstance(ret, dict):
                    output_attrs = cls._build_state_attributes(
                        ret, is_input=False
                    )
                    attributes.update(output_attrs)
                else:
                    attributes[SpanAttributes.GRAPH_NODE.OUTPUT_STATE] = str(
                        ret
                    )
            except Exception as e:
                attributes[SpanAttributes.GRAPH_NODE.OUTPUT_STATE] = (
                    f"Error serializing output: {str(e)}"
                )

        # Handle exceptions
        if exception:
            attributes[SpanAttributes.GRAPH_NODE.ERROR] = str(exception)

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
                        cls._update_span_name(task_name)

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
    def _create_node_update_attributes(
        cls, node_name: str, node_data: Any
    ) -> dict:
        """Create attributes for individual node updates in streaming scenarios.

        This follows the same pattern as task_function_attributes and pregel_attributes
        for consistency across the codebase.
        """
        attributes = {
            SpanAttributes.GRAPH_NODE.NODE_NAME: node_name,
        }

        if isinstance(node_data, dict):
            state_attrs = cls._build_state_attributes(node_data, is_input=False)
            attributes.update(state_attrs)
        else:
            attributes[SpanAttributes.GRAPH_NODE.OUTPUT_STATE] = str(node_data)

        return attributes

    @classmethod
    def _wrap_stream_generator(cls, original_generator):
        """Wrap a LangGraph stream generator to capture individual node updates as spans"""

        try:

            def instrumented_generator():
                for chunk in original_generator:
                    # Each chunk typically contains node updates
                    if isinstance(chunk, dict):
                        for node_name, node_data in chunk.items():
                            span_name = f"graph_node.{node_name}"

                            try:
                                with create_function_call_context_manager(
                                    create_new_span=True, span_name=span_name
                                ) as span:
                                    set_general_span_attributes(
                                        span, SpanAttributes.SpanType.GRAPH_NODE
                                    )

                                    attributes = (
                                        cls._create_node_update_attributes(
                                            node_name, node_data
                                        )
                                    )

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
                    original_generator = original_method(self, *args, **kwargs)

                    return cls._wrap_stream_generator(original_generator)

                setattr(
                    instrumented_method, TRULENS_INSTRUMENT_WRAPPER_FLAG, True
                )
                return instrumented_method

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
    def _setup_runnable_callable_instrumentation(cls):
        """Set up instrumentation for RunnableCallable objects (individual node functions)"""

        try:
            from langgraph.utils.runnable import RunnableCallable
        except ImportError:
            logger.warning(
                "RunnableCallable not available, skipping node instrumentation"
            )
            return

        if not is_otel_tracing_enabled():
            logger.debug(
                "OTEL not enabled, skipping RunnableCallable instrumentation"
            )
            return

        try:
            if hasattr(RunnableCallable, "invoke") and hasattr(
                getattr(RunnableCallable, "invoke"),
                TRULENS_INSTRUMENT_WRAPPER_FLAG,
            ):
                logger.debug("RunnableCallable methods already instrumented")
                return

            logger.info(
                "Applying class-level instrumentation to RunnableCallable methods"
            )

            def runnable_callable_attributes(ret, exception, *args, **kwargs):
                """Extract attributes for RunnableCallable (node function) execution"""
                attributes = {}

                # Get the function reference to extract the actual function name
                node_name = "unknown_node"
                if args and hasattr(args[0], "func"):
                    func = args[0].func
                    if hasattr(func, "__name__"):
                        node_name = func.__name__
                    else:
                        node_name = str(func)

                attributes[SpanAttributes.GRAPH_NODE.NODE_NAME] = node_name

                # Update the span name to the actual node function name
                cls._update_span_name(node_name)

                # Handle input/output state and exceptions
                input_data = args[1] if len(args) > 1 else None
                cls._handle_input_output_state(
                    input_data, ret, exception, attributes
                )

                return attributes

            # Apply custom instrumentation to RunnableCallable methods with filtering
            for method_name in ["invoke", "ainvoke"]:
                cls._instrument_runnable_callable_method(
                    RunnableCallable, method_name, runnable_callable_attributes
                )
                logger.debug(
                    f"Applied class-level instrumentation to RunnableCallable.{method_name}"
                )

        except Exception as e:
            logger.warning(f"Failed to instrument RunnableCallable: {e}")

    @classmethod
    def _instrument_runnable_callable_method(
        cls, target_class, method_name, attributes_func
    ):
        """Apply instrumentation to RunnableCallable method with filtering for internal nodes"""
        import wrapt

        original_method = getattr(target_class, method_name)

        def filtered_wrapper(wrapped, instance, args, kwargs):
            # Check if this is an internal LangGraph node before creating any span
            if hasattr(instance, "func") and hasattr(instance.func, "__name__"):
                func_name = instance.func.__name__
                # Skip instrumentation for internal LangGraph nodes
                if func_name.startswith("_") and func_name in [
                    "_write",
                    "_route",
                    "_control_branch",
                ]:
                    return wrapped(*args, **kwargs)

            # For user-defined nodes, apply full instrumentation
            instrumented_method = instrument(
                span_type=SpanAttributes.SpanType.GRAPH_NODE,
                attributes=attributes_func,
            )(wrapped)
            return instrumented_method(*args, **kwargs)

        # Apply the wrapper using wrapt
        setattr(
            target_class,
            method_name,
            wrapt.decorator(filtered_wrapper)(original_method),
        )

    @classmethod
    def _setup_pregel_instrumentation(cls):
        """Set up class-level instrumentation for Pregel methods"""

        if not is_otel_tracing_enabled():
            logger.debug(
                "OTEL not enabled, skipping Pregel class-level instrumentation"
            )
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

            # Set up RunnableCallable instrumentation for individual nodes
            cls._setup_runnable_callable_instrumentation()

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

                # Handle input from args and kwargs
                input_data = args[1] if args and len(args) > 1 else None

                # Also handle kwargs as potential input
                if kwargs:
                    for _, v in kwargs.items():
                        if isinstance(v, dict):
                            # If we haven't found input_data yet, use this dict
                            if input_data is None:
                                input_data = v
                            break

                # Handle streaming method output specially
                output_data = ret
                if method_name in [
                    "stream",
                    "astream",
                    "stream_mode",
                    "astream_mode",
                ]:
                    output_data = f"<Generator: {type(ret).__name__}>"

                # Use the consolidated helper
                cls._handle_input_output_state(
                    input_data, output_data, exception, attributes
                )

                # Update span name for Pregel methods
                span_name_for_pregel = (
                    "graph"  # user-friendly name instead of "Pregel.invoke"
                )
                cls._update_span_name(span_name_for_pregel)

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

            for method_name in streaming_methods:
                cls._instrument_streaming_method(method_name)
                logger.debug(
                    f"Applied streaming instrumentation to Pregel.{method_name}"
                )

            logger.debug(
                "Successfully applied class-level instrumentation to Pregel methods"
            )

        except Exception as e:
            logger.warning(
                f"Failed to apply class-level Pregel instrumentation: {e}"
            )

    @classmethod
    def _ensure_instrumentation(cls):
        """Ensure one-time initialization of instrumentation."""
        if not cls._is_instrumented:
            cls._setup_task_function_instrumentation()
            cls._setup_pregel_instrumentation()
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

    def _validate_pregel_app(self):
        """Validate that the app is a Pregel instance."""
        if not isinstance(self.app, Pregel):
            raise Exception(
                f"App must be an instance of Pregel, got {type(self.app)}"
            )

    def main_call(self, human: str):
        """A single text to a single text invocation of this app."""
        self._validate_pregel_app()
        try:
            result = self.app.invoke({"messages": [("user", human)]})
            return self._extract_output_from_result(result)
        except Exception as e:
            raise Exception(f"Error invoking Langgraph workflow: {str(e)}")

    async def main_acall(self, human: str):
        """A single text to a single text async invocation of this app."""
        self._validate_pregel_app()
        try:
            result = await self.app.ainvoke({"messages": [("user", human)]})
            return self._extract_output_from_result(result)
        except Exception as e:
            raise Exception(f"Error invoking Langgraph workflow: {str(e)}")

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
