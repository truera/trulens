"""LangGraph app instrumentation."""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
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
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils
from trulens.otel.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

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

# Import LangGraph func module components
try:
    from langgraph.func import TaskFunction
    from langgraph.func import task

    LANGGRAPH_FUNC_AVAILABLE = True
except ImportError:
    task = None
    TaskFunction = type("TaskFunction", (), {})
    LANGGRAPH_FUNC_AVAILABLE = False


def _langgraph_graph_span() -> Dict[str, Any]:
    """Create span configuration for LangGraph graph operations."""

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        attributes = {}

        # Try to extract graph input state
        if args and len(args) > 1:  # args[0] is self, args[1] might be input
            input_data = args[1]
            if isinstance(input_data, dict):
                # Serialize the input state for tracing
                attributes[SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE] = str(
                    input_data
                )
            else:
                attributes[SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE] = str(
                    input_data
                )

        # Handle keyword arguments for input
        for k, v in kwargs.items():
            if k in ["input", "state", "data"]:
                attributes[SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE] = str(v)
                break

        # Extract output state
        if ret is not None and not exception:
            attributes[SpanAttributes.LANGGRAPH_GRAPH.OUTPUT_STATE] = str(ret)

        # Handle errors
        if exception:
            attributes[SpanAttributes.LANGGRAPH_GRAPH.ERROR] = str(exception)

        return attributes

    return {
        "span_type": SpanAttributes.SpanType.LANGGRAPH_GRAPH,
        "attributes": _attributes,
    }


def _langgraph_task_span() -> Dict[str, Any]:
    """Create span configuration for LangGraph task operations."""

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        attributes = {}

        # Try to extract task input state
        if args and len(args) > 1:  # args[0] is usually self or context
            input_data = args[1] if len(args) > 1 else args[0]
            attributes[SpanAttributes.LANGGRAPH_TASK.INPUT_STATE] = str(
                input_data
            )

        # Handle keyword arguments for input
        for k, v in kwargs.items():
            if k in ["state", "data", "context"]:
                attributes[SpanAttributes.LANGGRAPH_TASK.INPUT_STATE] = str(v)
                break

        # Extract output state
        if ret is not None and not exception:
            attributes[SpanAttributes.LANGGRAPH_TASK.OUTPUT_STATE] = str(ret)

        # Handle errors
        if exception:
            attributes[SpanAttributes.LANGGRAPH_TASK.ERROR] = str(exception)

        return attributes

    return {
        "span_type": SpanAttributes.SpanType.LANGGRAPH_TASK,
        "attributes": _attributes,
    }


def _langgraph_entrypoint_span() -> Dict[str, Any]:
    """Create span configuration for LangGraph entrypoint operations."""

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        attributes = {}

        # Try to extract entrypoint input data
        if args and len(args) > 1:
            input_data = args[1]
            attributes[SpanAttributes.LANGGRAPH_ENTRYPOINT.INPUT_DATA] = str(
                input_data
            )

        # Handle keyword arguments for input
        for k, v in kwargs.items():
            if k in ["input", "data", "query"]:
                attributes[SpanAttributes.LANGGRAPH_ENTRYPOINT.INPUT_DATA] = (
                    str(v)
                )
                break

        # Extract output data
        if ret is not None and not exception:
            attributes[SpanAttributes.LANGGRAPH_ENTRYPOINT.OUTPUT_DATA] = str(
                ret
            )

        # Handle errors
        if exception:
            attributes[SpanAttributes.LANGGRAPH_ENTRYPOINT.ERROR] = str(
                exception
            )

        return attributes

    return {
        "span_type": SpanAttributes.SpanType.LANGGRAPH_ENTRYPOINT,
        "attributes": _attributes,
    }


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
                TaskFunction,
            }
            if LANGGRAPH_AVAILABLE
            else set()
        )
        """Classes to instrument."""

        METHODS: List[InstrumentedMethod] = []

        # Build methods list conditionally
        if LANGGRAPH_AVAILABLE:
            METHODS.extend([
                # Core LangGraph methods
                InstrumentedMethod(
                    "invoke",
                    Pregel,
                    **_langgraph_graph_span(),
                ),
                InstrumentedMethod(
                    "ainvoke",
                    Pregel,
                    **_langgraph_graph_span(),
                ),
                InstrumentedMethod(
                    "stream",
                    Pregel,
                    **_langgraph_graph_span(),
                ),
                InstrumentedMethod(
                    "astream",
                    Pregel,
                    **_langgraph_graph_span(),
                ),
                InstrumentedMethod(
                    "stream_mode",
                    Pregel,
                    **_langgraph_graph_span(),
                ),
                InstrumentedMethod(
                    "astream_mode",
                    Pregel,
                    **_langgraph_graph_span(),
                ),
            ])

        if LANGGRAPH_FUNC_AVAILABLE:
            METHODS.extend([
                # Task function methods
                InstrumentedMethod(
                    "__call__",
                    TaskFunction,
                    **_langgraph_task_span(),
                ),
                InstrumentedMethod(
                    "invoke",
                    TaskFunction,
                    **_langgraph_task_span(),
                ),
                InstrumentedMethod(
                    "result",
                    TaskFunction,
                    **_langgraph_task_span(),
                ),
            ])

        # Note: entrypoint is a decorator, not a class to instrument directly
        # Entrypoint functions will be instrumented through their underlying Pregel instances

        """Methods to instrument."""

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
    - `Pregel` (compiled graphs) and `StateGraph` (uncompiled graphs)
    - `@task` decorated functions from `langgraph.func`
    - `@entrypoint` decorated functions from `langgraph.func`

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

    # TODEP
    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)
    """The root callable of the wrapped app."""

    def __init__(
        self,
        app: Any,
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

        # TruGraph specific:
        kwargs["app"] = app

        # Extract connector if provided for TruSession creation
        connector = kwargs.get("connector")
        if connector is not None:
            TruSession(connector=connector)
        else:
            TruSession()

        if main_method is not None:
            kwargs["main_method"] = main_method
        kwargs["root_class"] = pyschema_utils.Class.of_object(app)

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

    def _detect_langgraph_components(self, app: Any) -> List[Tuple[str, Any]]:
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
                if isinstance(attr_value, (Pregel, StateGraph, TaskFunction)):
                    components.append((attr_name, attr_value))
                    logger.debug(f"Found LangGraph component: {attr_name}")
            except Exception:
                # Skip attributes that can't be accessed
                continue

        return components

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
                    # messages is not a list or is empty
                    return str(ret)
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
            else:
                # messages is not a list or is empty
                return str(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)


TruGraph.model_rebuild()
