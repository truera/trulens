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

        # Monkey-patch the @task decorator to automatically instrument functions
        self._patch_task_decorator()

    def _patch_task_decorator(self):
        """Monkey-patch the @task decorator to automatically instrument decorated functions."""
        if not LANGGRAPH_AVAILABLE:
            return

        try:
            import langgraph.func as langgraph_func

            if hasattr(langgraph_func.task, "_trulens_patched"):
                return

            # Store the original decorator
            original_task = langgraph_func.task

            def instrumented_task(*task_args, **task_kwargs):
                """Instrumented version of the @task decorator."""

                def decorator(func: Callable) -> Callable:
                    logger.debug(
                        f"@task decorator called with func: {func} (type: {type(func)})"
                    )

                    # First apply the original @task decorator to get a TaskFunction
                    task_function = original_task(*task_args, **task_kwargs)(
                        func
                    )
                    logger.debug(
                        f"Original @task returned: {type(task_function)}, has .func: {hasattr(task_function, 'func')}"
                    )

                    if is_otel_tracing_enabled():
                        logger.debug(
                            "OTel tracing is enabled, attempting instrumentation..."
                        )
                        try:
                            from trulens.core.otel.instrument import instrument

                            # Ensure we have a valid function
                            if not callable(func):
                                logger.warning(
                                    f"Expected callable but got {type(func)}: {func}"
                                )
                                return task_function

                            # Get function name safely
                            func_name = getattr(func, "__name__", str(func))
                            if not isinstance(func_name, str):
                                func_name = str(func_name)
                            logger.debug(f"Function name: {func_name}")

                            # Check if we got a TaskFunction with .func attribute
                            if hasattr(task_function, "func") and callable(
                                task_function.func
                            ):
                                logger.debug(
                                    "TaskFunction has .func attribute, instrumenting..."
                                )
                                # Instrument the underlying function inside TaskFunction
                                original_func = task_function.func
                                logger.debug(
                                    f"Original func type: {type(original_func)}"
                                )

                                # Create attributes callable that extracts task information
                                def task_attributes(ret, exc, *args, **kwargs):
                                    """Extract attributes from @task function calls."""
                                    attributes = {}
                                    try:
                                        # Extract function name and basic info
                                        attributes["task.function_name"] = (
                                            func_name
                                        )
                                        attributes["task.module"] = getattr(
                                            func, "__module__", "unknown"
                                        )

                                        # Try to extract meaningful arguments (avoiding LLM objects)
                                        import inspect

                                        try:
                                            sig = inspect.signature(
                                                original_func
                                            )
                                            bound_args = sig.bind_partial(
                                                *args, **kwargs
                                            )

                                            for (
                                                name,
                                                value,
                                            ) in bound_args.arguments.items():
                                                # Skip self and common non-serializable objects
                                                if name in ["self", "_self"]:
                                                    continue

                                                # Try to extract basic info, avoid complex objects
                                                try:
                                                    if isinstance(
                                                        value,
                                                        (
                                                            str,
                                                            int,
                                                            float,
                                                            bool,
                                                            type(None),
                                                        ),
                                                    ):
                                                        attributes[
                                                            f"task.args.{name}"
                                                        ] = str(value)
                                                    elif isinstance(
                                                        value, dict
                                                    ):
                                                        attributes[
                                                            f"task.args.{name}"
                                                        ] = str(value)[
                                                            :500
                                                        ]  # Truncate long values
                                                    elif (
                                                        hasattr(
                                                            value, "__dict__"
                                                        )
                                                        and len(str(value))
                                                        < 1000
                                                    ):
                                                        attributes[
                                                            f"task.args.{name}"
                                                        ] = str(value)[:500]
                                                    else:
                                                        attributes[
                                                            f"task.args.{name}"
                                                        ] = f"<{type(value).__name__}>"
                                                except Exception:
                                                    attributes[
                                                        f"task.args.{name}"
                                                    ] = f"<{type(value).__name__}>"
                                        except Exception as sig_exc:
                                            logger.debug(
                                                f"Failed to extract signature for {func_name}: {sig_exc}"
                                            )

                                        # Add return value info
                                        if ret is not None:
                                            try:
                                                if isinstance(
                                                    ret, (str, int, float, bool)
                                                ):
                                                    attributes[
                                                        "task.return"
                                                    ] = str(ret)
                                                else:
                                                    attributes[
                                                        "task.return"
                                                    ] = f"<{type(ret).__name__}>"
                                            except Exception:
                                                attributes["task.return"] = (
                                                    f"<{type(ret).__name__}>"
                                                )

                                        # Add exception info if present
                                        if exc and str(exc) != "No exception":
                                            attributes["task.exception"] = str(
                                                exc
                                            )
                                            attributes[
                                                "task.exception_type"
                                            ] = type(exc).__name__

                                    except Exception as attr_exc:
                                        logger.debug(
                                            f"Failed to extract @task attributes: {attr_exc}"
                                        )
                                        attributes[
                                            "task.attribute_extraction_error"
                                        ] = str(attr_exc)

                                    return attributes

                                # Apply the instrument decorator to the underlying function
                                span_type = f"TASK_{func_name.upper()}"
                                logger.debug(
                                    f"Applying instrument with span_type: {span_type}"
                                )
                                instrumented_func = instrument(
                                    span_type=span_type,
                                    attributes=task_attributes,
                                )(original_func)
                                logger.debug(
                                    f"Instrumented func type: {type(instrumented_func)}"
                                )

                                # Replace the TaskFunction's .func with the instrumented version
                                # This preserves the TaskFunction object while instrumenting its execution
                                task_function.func = instrumented_func
                                logger.debug(
                                    "Replaced task_function.func with instrumented version"
                                )

                                logger.debug(
                                    f"Successfully instrumented @task function: {func_name}"
                                )

                            else:
                                logger.warning(
                                    f"TaskFunction object doesn't have .func attribute or it's not callable: {type(task_function)}"
                                )

                        except ImportError:
                            logger.debug(
                                f"TruLens OTel instrumentation not available for @task function {getattr(func, '__name__', str(func))}"
                            )
                        except Exception as e:
                            func_name_safe = getattr(
                                func, "__name__", str(func)
                            )
                            logger.exception(
                                f"Failed to instrument @task function {func_name_safe}: {e}"
                            )
                    else:
                        # Traditional TruLens mode - just log that we found a @task function
                        func_name_safe = getattr(func, "__name__", str(func))
                        logger.debug(
                            f"Found @task function {func_name_safe} (traditional TruLens mode)"
                        )

                    # Always return the TaskFunction object (instrumented or not)
                    logger.debug(
                        f"Returning task_function: {type(task_function)}"
                    )
                    return task_function

                # Handle both @task and @task(...) usage patterns
                if (
                    len(task_args) == 1
                    and len(task_kwargs) == 0
                    and callable(task_args[0])
                ):
                    # Direct usage: @task
                    func = task_args[0]
                    # Call the decorator with no arguments (reset task_args/kwargs for original_task)
                    task_args = ()
                    task_kwargs = {}
                    return decorator(func)
                else:
                    # Parameterized usage: @task(...)
                    # Return the decorator that will be called with the function
                    return decorator

            # Replace the original task decorator
            langgraph_func.task = instrumented_task
            langgraph_func.task._trulens_patched = True

            logger.debug(
                "Successfully monkey-patched LangGraph @task decorator for TruLens instrumentation"
            )

        except ImportError:
            logger.debug(
                "LangGraph @task decorator not available for monkey-patching"
            )
        except Exception as e:
            logger.warning(
                f"Failed to monkey-patch LangGraph @task decorator: {e}"
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

    app: Union[Pregel, StateGraph]
    """The langgraph app to be instrumented."""

    root_callable: ClassVar[Optional[pyschema_utils.FunctionOrMethod]] = Field(
        None
    )
    """The root callable of the wrapped app."""

    def __init__(
        self,
        app: Union[Pregel, StateGraph],
        main_method: Optional[Callable] = None,
        **kwargs: Dict[str, Any],
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                f"LangGraph is not installed. Please install it with 'pip install langgraph' "
                f"to use TruGraph. Error: {E}"
            )

        # For LangGraph apps, we need to check if it's a compiled graph
        # compile if it's a StateGraph
        if isinstance(app, StateGraph):
            logger.warning(
                "Received uncompiled StateGraph. Compiling it for instrumentation. "
                "For better control, consider compiling the graph yourself before wrapping with TruGraph."
            )
            app = app.compile()

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
            # The main method for LangGraph apps is typically 'invoke' or 'run'
            if main_method is None:
                if hasattr(app, "invoke"):
                    main_method = app.invoke
                elif hasattr(app, "run"):
                    main_method = app.run
                else:
                    raise ValueError(
                        "LangGraph app must have 'invoke' or 'run' method for OTel tracing. "
                        "Alternatively, specify main_method explicitly."
                    )

            kwargs["main_method"] = main_method
            # Skip traditional instrumentation in OTel mode (but keep @task patching)
            # Note: We already created langgraph_instrument above for @task patching
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
