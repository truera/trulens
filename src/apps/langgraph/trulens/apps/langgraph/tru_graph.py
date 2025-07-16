"""LangGraph app instrumentation."""

from inspect import BoundArguments
from inspect import Signature
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
)

from pydantic import Field
from trulens.apps.langchain.tru_chain import TruChain
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.instruments import InstrumentedMethod
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils

logger = logging.getLogger(__name__)

# Import the instrument decorator for @task function instrumentation
try:
    from trulens.core.otel.instrument import instrument
    from trulens.otel.semconv.trace import SpanAttributes

    TRULENS_INSTRUMENT_AVAILABLE = True
except ImportError:
    instrument = None
    SpanAttributes = None
    TRULENS_INSTRUMENT_AVAILABLE = False

E = None
# LangGraph imports with optional import handling
try:
    from langgraph.graph import StateGraph
    from langgraph.pregel import Pregel
    from langgraph.types import Command

    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    # Create mock classes when langgraph is not available
    StateGraph = type("StateGraph", (), {})
    Pregel = type("Pregel", (), {})
    Command = type("Command", (), {})
    LANGGRAPH_AVAILABLE = False
    E = e

try:
    from langgraph.func import task

    LANGGRAPH_TASK_AVAILABLE = True
except ImportError:
    task = None
    LANGGRAPH_TASK_AVAILABLE = False

# Global flag to track if early patching has been done
_GLOBAL_TASK_PATCHING_DONE = False


def force_early_task_patching():
    """Force early @task monkey-patching before workflows are created."""
    global _GLOBAL_TASK_PATCHING_DONE

    if _GLOBAL_TASK_PATCHING_DONE:
        return

    if not LANGGRAPH_AVAILABLE or not TRULENS_INSTRUMENT_AVAILABLE:
        return

    try:
        # Create a temporary instrument to do the patching
        temp_instrument = LangGraphInstrument()  # noqa: F841
        _GLOBAL_TASK_PATCHING_DONE = True
        logger.info("Early @task monkey-patching completed successfully")
    except Exception as e:
        logger.warning(f"Early @task monkey-patching failed: {e}")


class LangGraphInstrument(core_instruments.Instrument):
    """Instrumentation for LangGraph apps."""

    class Default:
        """Instrumentation specification for LangGraph apps."""

        MODULES = {"langgraph"}
        """Filter for module name prefix for modules to be instrumented."""

        CLASSES = (
            lambda: {
                Pregel,
                StateGraph,
            }
            if LANGGRAPH_AVAILABLE
            else set()
        )
        """Filter for classes to be instrumented."""

        # Instrument only methods with these names and of these classes.
        METHODS: List[InstrumentedMethod] = (
            [
                InstrumentedMethod("invoke", Pregel),
                InstrumentedMethod("ainvoke", Pregel),
                InstrumentedMethod("stream", Pregel),
                InstrumentedMethod("astream", Pregel),
                InstrumentedMethod("stream_mode", Pregel),
                InstrumentedMethod("astream_mode", Pregel),
            ]
            if LANGGRAPH_AVAILABLE
            else []
        )
        """Methods to be instrumented.

        Key is method name and value is filter for objects that need those
        methods instrumented"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LangGraphInstrument.Default.MODULES,
            include_classes=LangGraphInstrument.Default.CLASSES(),
            include_methods=LangGraphInstrument.Default.METHODS,
            *args,
            **kwargs,
        )

        # Comprehensive @task monkey-patching
        self._comprehensive_task_patching()

    def _comprehensive_task_patching(self):
        """Apply working @task monkey-patching that actually works."""
        if not LANGGRAPH_AVAILABLE or not TRULENS_INSTRUMENT_AVAILABLE:
            logger.debug(
                "LangGraph or TruLens instrument not available for @task patching"
            )
            return

        try:
            # Import required modules
            import langgraph.func as langgraph_func
            from langgraph.func import task

            # Check if already patched with working fix
            if hasattr(task, "_trulens_working_patch"):
                logger.debug("@task decorator already patched with working fix")
                return

            # Store the original task decorator
            original_task = task

            def working_instrumented_task(*args, **kwargs):
                """Working instrumented version of @task decorator."""

                def decorator(func):
                    logger.debug(
                        f"Instrumenting @task function: {func.__name__}"
                    )

                    # Apply TruLens instrumentation FIRST
                    try:
                        instrumented_func = instrument(
                            span_type="langgraph_task",  # type: ignore
                            attributes=self._extract_task_attributes_simple,
                        )(func)
                        logger.debug(
                            f"Applied TruLens instrumentation to {func.__name__}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"TruLens instrumentation failed for {func.__name__}: {e}"
                        )
                        instrumented_func = func

                    # Then apply the original @task decorator
                    try:
                        if args or kwargs:
                            # Parameterized @task call: @task(...)
                            task_func = original_task(*args, **kwargs)(
                                instrumented_func
                            )
                        else:
                            # Direct @task call: @task
                            task_func = original_task(instrumented_func)

                        logger.debug(
                            f"Applied @task decorator to {func.__name__}"
                        )
                        return task_func
                    except Exception as e:
                        logger.warning(
                            f"@task decoration failed for {func.__name__}: {e}"
                        )
                        return instrumented_func

                # Handle both @task and @task(...) patterns
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    # Direct usage: @task
                    func = args[0]
                    return decorator(func)
                else:
                    # Parameterized usage: @task(...)
                    return decorator

            # Replace the task decorator
            langgraph_func.task = working_instrumented_task

            # Mark as patched with working fix
            langgraph_func.task._trulens_working_patch = True

            logger.info("Successfully applied working @task instrumentation")

        except Exception as e:
            logger.warning(f"Working @task instrumentation failed: {e}")

    def _extract_task_attributes_simple(
        self, ret: Any, exc: Optional[Exception], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Simple attribute extraction for @task functions.

        Args:
            ret: Return value from the function
            exc: Exception if any occurred
            *args: Positional arguments passed to the function
            **kwargs: Keyword arguments passed to the function

        Returns:
            Dictionary of attributes for tracing
        """
        attributes = {}

        try:
            # Get function name from the call stack
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                func_name = frame.f_back.f_back.f_code.co_name
                attributes["task.function_name"] = func_name

            # Return value info
            if ret is not None:
                attributes["task.return_type"] = type(ret).__name__
            else:
                attributes["task.return_type"] = "None"

            # Exception info
            attributes["task.has_exception"] = exc is not None
            if exc:
                attributes["task.exception_type"] = type(exc).__name__
            else:
                attributes["task.exception_type"] = None

            # Basic argument count (avoid complex serialization)
            attributes["task.arg_count"] = len(args)
            attributes["task.kwarg_count"] = len(kwargs)

        except Exception as e:
            logger.debug(f"Failed to extract simple task attributes: {e}")
            attributes["task.extraction_error"] = str(e)

        return attributes

    def _extract_task_attributes(
        self, ret: Any, exc: Optional[Exception], *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Extract attributes from @task function calls using simple heuristics.

        Args:
            ret: Return value from the function
            exc: Exception if any occurred
            *args: Positional arguments passed to the function
            **kwargs: Keyword arguments passed to the function

        Returns:
            Dictionary of attributes for tracing
        """
        attributes = {}

        try:
            # Basic function info
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                func_name = frame.f_back.f_back.f_code.co_name
                attributes["task.function_name"] = func_name

            # Simple heuristic: extract basic types and string representations
            # Skip args[0] if it looks like 'self'
            start_idx = 1 if args and hasattr(args[0], "__dict__") else 0

            for i, arg in enumerate(args[start_idx:], start_idx):
                if isinstance(arg, (str, int, float, bool, type(None))):
                    attributes[f"task.arg_{i}"] = str(arg)
                elif isinstance(arg, dict) and len(str(arg)) < 500:
                    attributes[f"task.arg_{i}"] = str(arg)[:500]
                else:
                    attributes[f"task.arg_{i}"] = f"<{type(arg).__name__}>"

            # Simple kwarg extraction
            for key, value in kwargs.items():
                if key in ["self", "_self"]:  # Skip self references
                    continue
                if isinstance(value, (str, int, float, bool, type(None))):
                    attributes[f"task.{key}"] = str(value)
                elif isinstance(value, dict) and len(str(value)) < 500:
                    attributes[f"task.{key}"] = str(value)[:500]
                else:
                    attributes[f"task.{key}"] = f"<{type(value).__name__}>"

            # Return value
            if ret is not None:
                if isinstance(ret, (str, int, float, bool)):
                    attributes["task.return"] = str(ret)
                elif isinstance(ret, dict) and len(str(ret)) < 500:
                    attributes["task.return"] = str(ret)[:500]
                else:
                    attributes["task.return"] = f"<{type(ret).__name__}>"

            # Exception info
            if exc:
                attributes["task.exception"] = str(exc)
                attributes["task.exception_type"] = type(exc).__name__

        except Exception as e:
            logger.debug(f"Failed to extract task attributes: {e}")
            attributes["task.extraction_error"] = str(e)

        return attributes


class TruGraph(TruChain):
    """Recorder for _LangGraph_ applications.

    This recorder is designed for LangGraph apps, providing a way to instrument,
    log, and evaluate their behavior while inheriting all LangChain instrumentation
    capabilities.

    **Flexible App Support**:

    `TruGraph` can wrap different types of objects:

    1. **Direct LangGraph objects**: `Pregel` (compiled graphs) or `StateGraph` (uncompiled)
    2. **Custom classes**: Any class that uses LangGraph workflows internally
    3. **Complex agents**: Multi-step applications with multiple LangGraph invocations

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

    root_callable: ClassVar[Optional[pyschema_utils.FunctionOrMethod]] = Field(
        None
    )
    """The root callable of the wrapped app."""

    def __init__(
        self,
        app: Any,  # Changed from Union[Pregel, StateGraph] to Any
        main_method: Optional[Callable] = None,
        **kwargs: Dict[str, Any],
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                f"LangGraph is not installed. Please install it with 'pip install langgraph' "
                f"to use TruGraph. Error: {E}"
            )

        # Force early @task monkey-patching before any workflows are processed
        force_early_task_patching()

        # Auto-detect and handle different app types
        app = self._prepare_app(app)

        kwargs["app"] = app

        if "connector" in kwargs:
            TruSession(connector=kwargs["connector"])
        else:
            TruSession()

        if main_method is not None:
            kwargs["main_method"] = main_method
        kwargs["root_class"] = pyschema_utils.Class.of_object(app)

        # Always ensure @task monkey-patching happens, regardless of instrumentation mode
        langgraph_instrument = LangGraphInstrument(app=self)  # noqa: F841

        if is_otel_tracing_enabled():
            # In OTel mode, set main_method for automatic instrumentation
            if main_method is None:
                main_method = self._detect_main_method(app)

            kwargs["main_method"] = main_method
        else:
            # Traditional instrumentation mode
            # Create combined instrumentation for both LangChain and LangGraph
            from trulens.apps.langchain.tru_chain import LangChainInstrument

            class CombinedInstrument(core_instruments.Instrument):
                def __init__(self, *args, **kwargs):
                    # Initialize with both LangChain and LangGraph settings
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

        # Call TruChain's parent (core_app.App) __init__ directly to avoid TruChain's specific initialization
        core_app.App.__init__(self, **kwargs)

    def _prepare_app(self, app: Any) -> Any:
        """
        Prepare the app for instrumentation by handling different input types.

        Args:
            app: The input application

        Returns:
            The prepared application ready for instrumentation
        """
        # Handle direct LangGraph objects
        if isinstance(app, StateGraph):
            logger.warning(
                "Received uncompiled StateGraph. Compiling it for instrumentation. "
                "For better control, consider compiling the graph yourself before wrapping with TruGraph."
            )
            return app.compile()

        if isinstance(app, Pregel):
            return app

        # Handle custom classes - simple detection for logging
        langgraph_components = self._detect_langgraph_components(app)
        if langgraph_components:
            logger.info(
                f"Detected {len(langgraph_components)} LangGraph component(s) in custom class {type(app).__name__}: "
                f"{[f'{name}: {type(comp).__name__}' for name, comp in langgraph_components]}"
            )

        # Return the app as-is for custom classes
        return app

    def _detect_langgraph_components(self, app: Any) -> List[tuple]:
        """
        Simple detection of LangGraph components in custom classes.

        This is primarily for logging/visibility. The main instrumentation
        works regardless of what this detects.

        Args:
            app: The application object to inspect

        Returns:
            List of (attribute_name, component) tuples for found LangGraph components
        """
        components = []

        if not hasattr(app, "__dict__"):
            return components

        # Simple approach: only check direct attributes
        for attr_name in dir(app):
            if attr_name.startswith("_"):
                continue

            try:
                attr_value = getattr(app, attr_name)
                if isinstance(attr_value, (Pregel, StateGraph)):
                    components.append((attr_name, attr_value))
                    logger.debug(f"Found LangGraph component: {attr_name}")
            except Exception:
                # Skip attributes that can't be accessed
                continue

        return components

    def _detect_main_method(self, app: Any) -> Optional[Callable]:
        """
        Detect the main method to instrument for the given app.

        Args:
            app: The application object

        Returns:
            The main method to instrument, or None if not found
        """
        # For direct LangGraph objects
        if isinstance(app, Pregel):
            if hasattr(app, "invoke"):
                return app.invoke
            elif hasattr(app, "run"):
                return app.run

        # For custom classes, look for common method names
        common_methods = ["run", "invoke", "execute", "call", "__call__"]

        for method_name in common_methods:
            if hasattr(app, method_name):
                method = getattr(app, method_name)
                if callable(method):
                    logger.info(f"Auto-detected main method: {method_name}")
                    return method

        # If no common methods found, raise an error with helpful message
        available_methods = [
            name
            for name in dir(app)
            if not name.startswith("_") and callable(getattr(app, name, None))
        ]

        raise ValueError(
            f"Could not auto-detect main method for {type(app).__name__}. "
            f"Auto-detection only works for common method names: {', '.join(['run', 'invoke', 'execute', 'call', '__call__'])}. "
            f"Available methods: {available_methods}. "
            f"For complex applications, please specify main_method explicitly: "
            f"TruGraph(app, main_method=app.your_main_method)"
        )

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """
        # For LangGraph, the main input is typically the initial state
        # which can be a dict with "messages" key or direct input
        if "input" in bindings.arguments:
            temp = bindings.arguments["input"]
            if isinstance(temp, dict):
                # For LangGraph, common patterns are:
                # {"messages": [HumanMessage(content="...")]}
                # or {"query": "..."}
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
                    # Try to get any string-like value from the input dict
                    for _, value in temp.items():
                        if isinstance(value, str):
                            return value
                    return str(temp)
            elif isinstance(temp, str):
                return temp
            else:
                return str(temp)

        # Fall back to TruChain's main_input method
        return super().main_input(func, sig, bindings)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> str:
        """
        Determine the main output string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """
        # For LangGraph, the output is typically the final state
        # which can be a dict with "messages" key
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
                # Try to get any string-like value from the output dict
                for _, value in ret.items():
                    if isinstance(value, str):
                        return value
                return str(ret)
        elif isinstance(ret, str):
            return ret
        else:
            # Fall back to TruChain's main_output method
            return super().main_output(func, sig, bindings, ret)

    def main_call(self, human: str):
        """
        A single text to a single text invocation of this app.
        """
        # Most LangGraph apps expect a dict with "messages" key
        try:
            # Try the common LangGraph pattern first
            result = self.app.invoke({"messages": [("user", human)]})
            return self._extract_output_from_result(result)
        except Exception:
            try:
                result = self.app.invoke(human)
                return self._extract_output_from_result(result)
            except Exception:
                return super().main_call(human)

    async def main_acall(self, human: str):
        """
        A single text to a single text async invocation of this app.
        """
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
        """
        Helper method to extract string output from LangGraph result.
        """
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
        elif isinstance(result, str):
            return result
        else:
            return str(result)


TruGraph.model_rebuild()
