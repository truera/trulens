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
    Tuple,
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

# Import TaskFunction for detection
try:
    from langgraph.func import TaskFunction

    TASK_FUNCTION_AVAILABLE = True
except ImportError:
    TaskFunction = None
    TASK_FUNCTION_AVAILABLE = False


def _detect_task_methods(app_instance: Any) -> List[Tuple[str, Any, Callable]]:
    """
    Detect @task decorated methods on an app instance.

    Args:
        app_instance: The application instance to inspect

    Returns:
        List of tuples (method_name, task_function_instance, original_function)
    """
    task_methods = []

    if app_instance is None:
        return task_methods

    logger.debug(f"Scanning {type(app_instance).__name__} for @task methods...")

    # Scan all attributes of the instance
    for attr_name in dir(app_instance):
        if attr_name.startswith("_"):
            continue

        try:
            attr_value = getattr(app_instance, attr_name)

            # Check if it's a TaskFunction (from @task decorator)
            if (
                TASK_FUNCTION_AVAILABLE
                and TaskFunction is not None
                and isinstance(attr_value, TaskFunction)
            ):
                task_methods.append((attr_name, attr_value, attr_value.func))
                logger.info(f"Found @task method: {attr_name}")
            # Fallback: check by class name for compatibility
            elif (
                hasattr(attr_value, "__class__")
                and attr_value.__class__.__name__ == "TaskFunction"
            ):
                original_func = getattr(attr_value, "func", None)
                if original_func:
                    task_methods.append((attr_name, attr_value, original_func))
                    logger.info(f"Found @task method (fallback): {attr_name}")

        except Exception as e:
            logger.debug(f"Error inspecting attribute {attr_name}: {e}")
            continue

    logger.info(
        f"Detected {len(task_methods)} @task methods on {type(app_instance).__name__}"
    )
    return task_methods


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
        # Get the app instance from kwargs to detect @task methods
        app_instance = kwargs.get("app")
        if hasattr(app_instance, "app"):
            # If it's a TruGraph instance, get the underlying app
            actual_app = getattr(app_instance, "app")
        else:
            actual_app = app_instance

        # Start with default methods
        methods_to_instrument = list(LangGraphInstrument.Default.METHODS)
        classes_to_instrument = LangGraphInstrument.Default.CLASSES()

        # Detect and add @task methods for instrumentation
        if actual_app is not None:
            task_methods = _detect_task_methods(actual_app)

            if task_methods:
                logger.info(
                    f"Adding {len(task_methods)} @task methods to instrumentation"
                )

                # Add the app's class to classes to instrument
                app_class = type(actual_app)
                classes_to_instrument.add(app_class)

                # Add each @task method to the instrumentation list
                for method_name, method_obj, original_func in task_methods:
                    instrumented_method = InstrumentedMethod(
                        method_name, app_class
                    )
                    methods_to_instrument.append(instrumented_method)
                    logger.info(
                        f"Added @task method to instrumentation: {method_name}"
                    )

        super().__init__(
            include_modules=LangGraphInstrument.Default.MODULES,
            include_classes=classes_to_instrument,
            include_methods=methods_to_instrument,
            *args,
            **kwargs,
        )


class TruGraph(TruChain):
    """Recorder for _LangGraph_ applications.

    This recorder is designed for LangGraph apps, providing a way to instrument,
    log, and evaluate their behavior while inheriting all LangChain instrumentation
    capabilities.

    **Automatic @task Function Tracing**:

    TruGraph automatically instruments all LangGraph `@task` decorated functions
    by detecting existing TaskFunction instances on the app object. No code changes
    are required - simply use standard LangGraph patterns:

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

        # Create LangGraph instrumentation
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

# Note: @task instrumentation is applied through detection when TruGraph is instantiated
