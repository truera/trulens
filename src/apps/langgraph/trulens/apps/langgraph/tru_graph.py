"""LangGraph app instrumentation."""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import (
    Any,
    Callable,
    ClassVar,
    List,
    Optional,
    Tuple,
)

from pydantic import Field
from trulens.apps.langchain.tru_chain import TruChain
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.instruments import InstrumentedMethod
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

E = None
try:
    from langgraph.graph import StateGraph
    from langgraph.pregel import Pregel
    from langgraph.types import Command

    LANGGRAPH_AVAILABLE = True
except ImportError as e:

    def _mock_compile(self):
        raise ImportError("LangGraph is not available")

    StateGraph = type("StateGraph", (), {"compile": _mock_compile})
    Pregel = type("Pregel", (), {})
    Command = type("Command", (), {})
    LANGGRAPH_AVAILABLE = False
    E = e

try:
    from langgraph.func import TaskFunction
    from langgraph.func import entrypoint
    from langgraph.func import task

    LANGGRAPH_FUNC_AVAILABLE = True

    # Apply class-level instrumentation for TaskFunction when langgraph.func is available
    # This ensures TaskFunction.__call__ is instrumented regardless of where TaskFunction
    # instances end up in the object hierarchy (e.g., embedded in Pregel workflows)
    def _setup_task_function_instrumentation():
        """Set up class-level instrumentation for TaskFunction.__call__"""
        import os

        # Only instrument if OTEL tracing is enabled
        otel_enabled = os.environ.get("TRULENS_OTEL_TRACING", "0") == "1"
        if not otel_enabled:
            logger.debug(
                "OTEL not enabled, skipping TaskFunction class-level instrumentation"
            )
            return

        try:
            from trulens.core.otel.instrument import instrument_method

            # Check if TaskFunction.__call__ is already instrumented
            if hasattr(TaskFunction.__call__, "__trulens_instrument_wrapper__"):
                logger.debug("TaskFunction.__call__ already instrumented")
                return

            logger.info(
                "Applying class-level instrumentation to TaskFunction.__call__"
            )

            # Create attributes function for TaskFunction calls
            def task_function_attributes(ret, exception, *args, **kwargs):
                attributes = {}

                # For TaskFunction.__call__, the first argument is self (TaskFunction instance)
                if args and len(args) > 0:
                    task_function_instance = args[0]
                    task_args = args[1:] if len(args) > 1 else ()
                    task_kwargs = kwargs

                    # Get the original function name
                    if hasattr(task_function_instance, "func") and hasattr(
                        task_function_instance.func, "__name__"
                    ):
                        attributes[SpanAttributes.LANGGRAPH_TASK.TASK_NAME] = (
                            task_function_instance.func.__name__
                        )

                    # Serialize the task input arguments
                    try:
                        import inspect
                        import json

                        if hasattr(task_function_instance, "func"):
                            # Try to bind arguments, but be robust about mismatches
                            try:
                                sig = inspect.signature(
                                    task_function_instance.func
                                )
                                # Be more careful about binding - only bind positional args that fit
                                # and exclude kwargs that don't match the signature

                                # Filter kwargs to only include those that match the signature
                                sig_params = list(sig.parameters.keys())
                                filtered_kwargs = {
                                    k: v
                                    for k, v in task_kwargs.items()
                                    if k in sig_params
                                }

                                # Try to bind with available arguments
                                if task_args or filtered_kwargs:
                                    bound_args = sig.bind_partial(
                                        *task_args, **filtered_kwargs
                                    )
                                    bound_args.apply_defaults()

                                    # Collect all arguments into a single JSON structure
                                    input_args = {}
                                    for (
                                        name,
                                        value,
                                    ) in bound_args.arguments.items():
                                        # Skip complex objects like models/pools
                                        if hasattr(value, "__module__") and (
                                            "llm" in str(type(value)).lower()
                                            or "pool"
                                            in str(type(value)).lower()
                                        ):
                                            continue

                                        try:
                                            if hasattr(
                                                value, "__dict__"
                                            ) and hasattr(
                                                value, "__dataclass_fields__"
                                            ):
                                                import dataclasses

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

                                    # Store as single JSON structure using proper SpanAttributes
                                    attributes[
                                        SpanAttributes.LANGGRAPH_TASK.INPUT_STATE
                                    ] = json.dumps(
                                        input_args, default=str, indent=2
                                    )
                                else:
                                    # No arguments to bind
                                    attributes[
                                        SpanAttributes.LANGGRAPH_TASK.INPUT_STATE
                                    ] = "{}"

                            except Exception as bind_error:
                                # If binding fails, fall back to simple argument logging
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
                                    SpanAttributes.LANGGRAPH_TASK.INPUT_STATE
                                ] = json.dumps(
                                    fallback_args, default=str, indent=2
                                )
                        else:
                            # Fallback: just stringify the arguments
                            attributes[
                                SpanAttributes.LANGGRAPH_TASK.INPUT_STATE
                            ] = json.dumps(
                                {"args": task_args, "kwargs": task_kwargs},
                                default=str,
                                indent=2,
                            )

                    except Exception as e:
                        logger.warning(
                            f"Error processing task input arguments: {e}"
                        )
                        # Even more basic fallback
                        attributes[
                            SpanAttributes.LANGGRAPH_TASK.INPUT_STATE
                        ] = f"Error processing args: {str(e)}"

                # Handle return value (Future object for now)
                if ret is not None and not exception:
                    try:
                        # For now, capture the Future object info
                        # TODO: In future, we might want to capture the actual result when Future completes
                        attributes[
                            SpanAttributes.LANGGRAPH_TASK.OUTPUT_STATE
                        ] = str(ret)
                    except Exception:
                        attributes[
                            SpanAttributes.LANGGRAPH_TASK.OUTPUT_STATE
                        ] = str(ret)

                if exception:
                    attributes[SpanAttributes.LANGGRAPH_TASK.ERROR] = str(
                        exception
                    )

                return attributes

            # Apply the instrumentation at class level
            instrument_method(
                cls=TaskFunction,
                method_name="__call__",
                span_type=SpanAttributes.SpanType.LANGGRAPH_TASK,
                attributes=task_function_attributes,
            )

            logger.info(
                "Successfully applied class-level instrumentation to TaskFunction.__call__"
            )

        except Exception as e:
            logger.warning(
                f"Failed to apply class-level TaskFunction instrumentation: {e}"
            )

    # Apply the instrumentation when this module is imported
    _setup_task_function_instrumentation()

    def _setup_pregel_instrumentation():
        """Set up class-level instrumentation for Pregel methods"""
        import os

        # Only instrument if OTEL tracing is enabled
        otel_enabled = os.environ.get("TRULENS_OTEL_TRACING", "0") == "1"
        if not otel_enabled:
            logger.debug(
                "OTEL not enabled, skipping Pregel class-level instrumentation"
            )
            return

        try:
            from trulens.core.otel.instrument import instrument_method

            # Check if Pregel methods are already instrumented
            if hasattr(Pregel, "invoke") and hasattr(
                getattr(Pregel, "invoke"), "__trulens_instrument_wrapper__"
            ):
                logger.debug("Pregel methods already instrumented")
                return

            logger.info(
                "Applying class-level instrumentation to Pregel methods"
            )

            # Create attributes function for Pregel methods
            def pregel_attributes(ret, exception, *args, **kwargs):
                attributes = {}

                if args and len(args) > 1:
                    input_data = args[1]
                    if isinstance(input_data, dict):
                        attributes[
                            SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE
                        ] = str(input_data)
                    else:
                        attributes[
                            SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE
                        ] = str(input_data)

                for k, v in kwargs.items():
                    if k in ["input", "state", "data"]:
                        attributes[
                            SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE
                        ] = str(v)
                        break

                if ret is not None and not exception:
                    attributes[SpanAttributes.LANGGRAPH_GRAPH.OUTPUT_STATE] = (
                        str(ret)
                    )

                if exception:
                    attributes[SpanAttributes.LANGGRAPH_GRAPH.ERROR] = str(
                        exception
                    )

                return attributes

            # Apply instrumentation to all Pregel methods
            pregel_methods = [
                "invoke",
                "ainvoke",
                "stream",
                "astream",
                "stream_mode",
                "astream_mode",
            ]
            for method_name in pregel_methods:
                instrument_method(
                    cls=Pregel,
                    method_name=method_name,
                    span_type=SpanAttributes.SpanType.LANGGRAPH_GRAPH,
                    attributes=pregel_attributes,
                )
                logger.debug(
                    f"Applied class-level instrumentation to Pregel.{method_name}"
                )

            logger.info(
                "Successfully applied class-level instrumentation to Pregel methods"
            )

        except Exception as e:
            logger.warning(
                f"Failed to apply class-level Pregel instrumentation: {e}"
            )

    # Apply Pregel instrumentation when this module is imported
    _setup_pregel_instrumentation()

except ImportError:
    task = None
    TaskFunction = type("TaskFunction", (), {})
    entrypoint = type("entrypoint", (), {})
    LANGGRAPH_FUNC_AVAILABLE = False


class LangGraphInstrument(core_instruments.Instrument):
    """Instrumentation for LangGraph apps."""

    class Default:
        """Instrumentation specification for LangGraph apps."""

        MODULES = {"langgraph"}
        """Modules by prefix to instrument."""

        CLASSES = (
            lambda: {
                Pregel,
                StateGraph,
                # Note: TaskFunction is instrumented at class-level during import,
                # @entrypoint returns Pregel objects which are already instrumented
            }
            if LANGGRAPH_AVAILABLE and LANGGRAPH_FUNC_AVAILABLE
            else {
                Pregel,
                StateGraph,
            }
            if LANGGRAPH_AVAILABLE
            else set()
        )
        """Classes to instrument."""

        METHODS: List[InstrumentedMethod] = []

        # Note: Both TaskFunction and Pregel methods are instrumented at class-level
        # during import via _setup_task_function_instrumentation() and
        # _setup_pregel_instrumentation(), not through instance-based instrumentation

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
    - `TaskFunction.__call__` method - instrumented at class-level during import
    - `StateGraph` objects (uncompiled graphs) for logging/debugging purposes

    **Class-Level Instrumentation**: Both `@task` functions (TaskFunction) and
    `Pregel` graph methods are instrumented at the class level when TruGraph is imported.
    This ensures all function calls are captured regardless of where the instances
    are embedded in the object hierarchy (e.g., inside custom classes).

    **Benefits of Class-Level Approach**:
    - **Guaranteed Coverage**: All TaskFunction and Pregel method calls are captured
    - **No Import Timing Issues**: Works regardless of when objects are created
    - **Consistent Span Types**: Properly sets "langgraph_task" and "langgraph_graph" span types

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
            # Your task logic here - automatically instrumented
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

    app: Any
    """The application to be instrumented. Can be LangGraph objects or custom classes."""

    # Fix the root_callable field to have the correct default
    root_callable: ClassVar[Optional[pyschema_utils.FunctionOrMethod]] = Field(
        default=None
    )
    """The root callable of the wrapped app."""

    def __init__(
        self,
        app: Any,
        main_method: Optional[Callable] = None,
        **kwargs: Any,
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                f"LangGraph is not installed. Please install it with 'pip install langgraph' "
                f"to use TruGraph. Error: {E}"
            )

        # Only do minimal preparation to avoid interfering with existing instrumentation
        original_app = app
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

        # Store original app for debugging
        self._original_app = original_app

        core_app.App.__init__(self, **kwargs)

        # Note: Nested Pregel instrumentation is now done lazily to avoid hanging during init
        # Call instrument_nested_pregel_objects() manually if needed for nested workflows

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
            return app.compile()  # type: ignore

        if isinstance(app, Pregel):
            return app

        # For custom classes, detect components for logging/debugging
        langgraph_components = self._detect_langgraph_components(app)
        if langgraph_components:
            logger.info(
                f"Detected {len(langgraph_components)} LangGraph component(s) in custom class {type(app).__name__}: "
                f"{[f'{name}: {type(comp).__name__}' for name, comp in langgraph_components]}"
            )

        return app

    def _detect_langgraph_components(self, app: Any) -> List[Tuple[str, Any]]:
        """
        Simple detection of LangGraph components in custom classes.

        This method looks for basic LangGraph components for logging/debugging purposes.
        Note: TaskFunction instances are instrumented at class-level during import
        and don't need to be detected here.

        Args:
            app: The application object to inspect

        Returns:
            List of (attribute_name, component) tuples for found LangGraph components
        """
        components = []

        if not hasattr(app, "__dict__"):
            return components

        # Check for basic LangGraph objects - Pregel and StateGraph
        # TaskFunction is instrumented at class-level and doesn't need detection
        for attr_name in dir(app):
            if attr_name.startswith("_"):
                continue

            try:
                attr_value = getattr(app, attr_name)
                if isinstance(attr_value, (Pregel, StateGraph)):
                    components.append((attr_name, attr_value))
                    logger.debug(
                        f"Found LangGraph component: {attr_name} ({type(attr_value).__name__})"
                    )
            except Exception:
                continue

        return components

    def instrument_nested_pregel_objects(self) -> None:
        """
        Manually instrument nested Pregel objects found in the app.

        This method helps the instrumentation system find and instrument nested Pregel objects
        (compiled LangGraph workflows) that might be nested within custom classes.

        This is OPTIONAL and only needed if:
        1. You have custom classes with nested Pregel workflows
        2. You want those nested workflows to be instrumented
        3. The automatic instrumentation didn't catch them

        Example:
            ```python
            class MyAgent:
                def __init__(self):
                    self.workflow = some_pregel_workflow

            agent = MyAgent()
            tru_graph = TruGraph(agent, main_method=agent.run)

            # Optionally instrument nested workflows
            tru_graph.instrument_nested_pregel_objects()
            ```
        """
        if not hasattr(self, "instrument") or not hasattr(
            self.instrument, "instrument_object"
        ):
            logger.debug(
                "No instrument object available for nested Pregel instrumentation"
            )
            return

        # Look for nested Pregel objects in the app
        nested_pregels = self._find_nested_pregel_objects(self.app)

        if not nested_pregels:
            logger.debug("No nested Pregel objects found")
            return

        logger.info(
            f"Found {len(nested_pregels)} nested Pregel objects to instrument"
        )

        for path_parts, pregel_obj in nested_pregels:
            try:
                # Create a lens path for this Pregel object
                lens = serial_utils.Lens()
                for part in path_parts:
                    lens = lens[part]

                logger.debug(
                    f"Instrumenting nested Pregel at path: {'.'.join(path_parts)}"
                )
                self.instrument.instrument_object(pregel_obj, lens)  # type: ignore

            except Exception as e:
                logger.warning(
                    f"Failed to instrument nested Pregel at {'.'.join(path_parts)}: {e}"
                )

    def _find_nested_pregel_objects(
        self, obj: Any, path: Optional[List[str]] = None, max_depth: int = 8
    ) -> List[Tuple[List[str], Any]]:
        """
        Recursively find Pregel objects nested within the given object.

        Args:
            obj: Object to search within
            path: Current path to this object
            max_depth: Maximum recursion depth to prevent hanging

        Returns:
            List of (path_parts, pregel_object) tuples
        """
        if path is None:
            path = []

        # Prevent excessive recursion that could cause hanging
        if len(path) >= max_depth:
            return []

        pregels = []

        # If this object itself is a Pregel, add it (but skip if it's the root app)
        if (
            isinstance(obj, Pregel) and path
        ):  # path check ensures we skip root app
            pregels.append((path, obj))
            return pregels  # Don't recurse into Pregel objects

        # Only traverse objects that are safe to traverse
        if not hasattr(obj, "__dict__"):
            return pregels

        # Be more selective about which attributes to check
        try:
            # Use __dict__ directly instead of dir() to avoid triggering properties
            obj_dict = getattr(obj, "__dict__", {})
            if not isinstance(obj_dict, dict):
                return pregels

            for attr_name, attr_value in obj_dict.items():
                # Skip private attributes and known problematic ones
                if attr_name.startswith("_") or attr_name in (
                    "session",
                    "connector",
                    "instrument",
                    "logger",
                ):
                    continue

                try:
                    if isinstance(attr_value, Pregel):
                        pregels.append((path + [attr_name], attr_value))
                    elif (
                        hasattr(attr_value, "__dict__")
                        and not callable(attr_value)
                        and not isinstance(
                            attr_value,
                            (str, int, float, bool, list, dict, tuple),
                        )
                    ):
                        # Only recurse into custom objects, avoid built-in types
                        nested_pregels = self._find_nested_pregel_objects(
                            attr_value, path + [attr_name], max_depth
                        )
                        pregels.extend(nested_pregels)

                except Exception:
                    # Silently skip any problematic attributes
                    continue

        except Exception:
            # If we can't safely traverse this object, skip it
            pass

        return pregels

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
        try:
            result = self.app.invoke({"messages": [("user", human)]})
            return self._extract_output_from_result(result)
        except Exception:
            try:
                result = self.app.invoke(human)
                return self._extract_output_from_result(result)
            except Exception:
                return super().main_call(human)

    async def main_acall(self, human: str):
        """A single text to a single text async invocation of this app."""
        try:
            result = await self.app.ainvoke({"messages": [("user", human)]})
            return self._extract_output_from_result(result)
        except Exception:
            try:
                result = await self.app.ainvoke(human)
                return self._extract_output_from_result(result)
            except Exception:
                return await super().main_acall(human)

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
