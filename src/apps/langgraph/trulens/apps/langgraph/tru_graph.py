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
    Union,
)

from pydantic import Field
from trulens.apps.langchain.tru_chain import TruChain
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.instruments import InstrumentedMethod
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils

logger = logging.getLogger(__name__)

# LangGraph imports with optional import handling
try:
    from langgraph.graph import CompiledStateGraph
    from langgraph.graph import StateGraph
    from langgraph.graph.state import StateDefinition
    from langgraph.pregel import Pregel
    from langgraph.types import Command

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Create mock classes when langgraph is not available
    StateGraph = type("StateGraph", (), {})
    CompiledStateGraph = type("CompiledStateGraph", (), {})
    Pregel = type("Pregel", (), {})
    Command = type("Command", (), {})
    StateDefinition = type("StateDefinition", (), {})
    LANGGRAPH_AVAILABLE = False

try:
    from langgraph.func import task

    LANGGRAPH_TASK_AVAILABLE = True
except ImportError:
    task = None
    LANGGRAPH_TASK_AVAILABLE = False


class LangGraphInstrument(core_instruments.Instrument):
    """Instrumentation for LangGraph apps."""

    class Default:
        """Instrumentation specification for LangGraph apps."""

        MODULES = {"langgraph"}
        """Filter for module name prefix for modules to be instrumented."""

        CLASSES = (
            lambda: {
                CompiledStateGraph,
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
                InstrumentedMethod("invoke", CompiledStateGraph),
                InstrumentedMethod("ainvoke", CompiledStateGraph),
                InstrumentedMethod("stream", CompiledStateGraph),
                InstrumentedMethod("astream", CompiledStateGraph),
                InstrumentedMethod("stream_mode", CompiledStateGraph),
                InstrumentedMethod("astream_mode", CompiledStateGraph),
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


class TruGraph(TruChain):
    """Recorder for _LangGraph_ applications.

    This recorder is designed for LangGraph apps, providing a way to instrument,
    log, and evaluate their behavior while inheriting all LangChain instrumentation
    capabilities.

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
        app: A LangGraph application (compiled StateGraph).

        **kwargs: Additional arguments to pass to [App][trulens.core.app.App]
            and [AppDefinition][trulens.core.schema.app.AppDefinition].
    """

    app: Union[CompiledStateGraph, Pregel, StateGraph]
    """The langgraph app to be instrumented."""

    # TODEP
    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)
    """The root callable of the wrapped app."""

    def __init__(
        self,
        app: Union[CompiledStateGraph, Pregel, StateGraph],
        main_method: Optional[Callable] = None,
        **kwargs: Dict[str, Any],
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not installed. Please install it with 'pip install langgraph' "
                "to use TruGraph."
            )

        # For LangGraph apps, we need to check if it's a compiled graph
        # If it's a StateGraph, we should compile it
        if isinstance(app, StateGraph):
            logger.warning(
                "Received uncompiled StateGraph. Compiling it for instrumentation. "
                "For better control, consider compiling the graph yourself before wrapping with TruGraph."
            )
            app = app.compile()

        # TruGraph specific:
        kwargs["app"] = app

        # Create `TruSession` if not already created.
        if "connector" in kwargs:
            TruSession(connector=kwargs["connector"])
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

                # Combine modules, classes, and methods
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
                    for key, value in temp.items():
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
                for key, value in ret.items():
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
        # For LangGraph, we need to format the input appropriately
        # Most LangGraph apps expect a dict with "messages" key
        try:
            # Try the common LangGraph pattern first
            result = self.app.invoke({"messages": [("user", human)]})
            return self._extract_output_from_result(result)
        except Exception:
            # Fall back to direct string input
            try:
                result = self.app.invoke(human)
                return self._extract_output_from_result(result)
            except Exception:
                # Fall back to the parent method
                return super().main_call(human)

    async def main_acall(self, human: str):
        """
        A single text to a single text async invocation of this app.
        """
        # For LangGraph, we need to format the input appropriately
        # Most LangGraph apps expect a dict with "messages" key
        try:
            # Try the common LangGraph pattern first
            result = await self.app.ainvoke({"messages": [("user", human)]})
            return self._extract_output_from_result(result)
        except Exception:
            # Fall back to direct string input
            try:
                result = await self.app.ainvoke(human)
                return self._extract_output_from_result(result)
            except Exception:
                # Fall back to the parent method
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


# Function to instrument @task decorators
def instrument_task(
    _func=None,
    *,
    span_type: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Decorator to instrument LangGraph @task functions.

    This decorator can be used to automatically instrument functions decorated with @task
    from LangGraph, providing tracing and evaluation capabilities.

    Args:
        span_type: Optional span type for the instrumentation
        attributes: Optional attributes to add to the span
        **kwargs: Additional arguments passed to the instrument decorator

    Example:
        ```python
        from trulens.apps.langgraph import instrument_task
        from langgraph.func import task

        @task
        @instrument_task(span_type="TASK_NODE")
        def my_task_function(state):
            # Task logic here
            return state
        ```
    """

    def decorator(func):
        if not LANGGRAPH_AVAILABLE:
            logger.warning(
                "LangGraph is not available. Task instrumentation will be skipped."
            )
            return func

        # Import the instrument decorator from trulens
        from trulens.core.otel.instrument import instrument

        # Set default span type if not provided
        if span_type is None:
            func_span_type = f"TASK_{func.__name__.upper()}"
        else:
            func_span_type = span_type

        # Apply the instrument decorator
        return instrument(
            span_type=func_span_type, attributes=attributes, **kwargs
        )(func)

    if _func is None:
        return decorator
    else:
        return decorator(_func)


TruGraph.model_rebuild()
