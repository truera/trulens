"""LangGraph app instrumentation."""

import functools
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
                    task_decorated_func = original_task(
                        *task_args, **task_kwargs
                    )(func)

                    try:
                        from trulens.core.otel.instrument import instrument

                        # Create a wrapper that extracts attributes using our heuristics
                        @functools.wraps(func)
                        def trulens_wrapper(*args, **kwargs):
                            ret = None
                            exc = None

                            try:
                                ret = task_decorated_func(*args, **kwargs)
                                return ret
                            except Exception as e:
                                exc = e
                                raise
                            finally:
                                # Extract attributes using our heuristics
                                try:
                                    if exc is not None:
                                        attributes = _extract_task_attributes(
                                            func, ret, exc, *args, **kwargs
                                        )
                                    else:
                                        attributes = _extract_task_attributes(
                                            func,
                                            ret,
                                            Exception("No exception"),
                                            *args,
                                            **kwargs,
                                        )

                                    logger.debug(
                                        f"Extracted @task attributes for {func.__name__}: {list(attributes.keys())}"
                                    )
                                except Exception as attr_exc:
                                    logger.debug(
                                        f"Failed to extract @task attributes for {func.__name__}: {attr_exc}"
                                    )

                        # Apply the instrument decorator to get full TruLens tracing
                        instrumented_func = instrument(
                            span_type=f"TASK_{func.__name__.upper()}"
                        )(trulens_wrapper)

                        logger.debug(
                            f"Successfully instrumented @task function: {func.__name__}"
                        )
                        return instrumented_func

                    except ImportError:
                        logger.debug(
                            f"TruLens instrumentation not available for @task function {func.__name__}"
                        )
                        return task_decorated_func

                return decorator

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

        # Check if OTel tracing is enabled
        from trulens.core.otel.utils import is_otel_tracing_enabled

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
            # Skip traditional instrumentation in OTel mode
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


# Helper function to extract attributes from @task functions
def _extract_task_attributes(
    func: Callable,
    ret: Any,
    exc: Exception,
    ignore_args: Optional[set] = None,
    extract_fields: Optional[Dict[str, str]] = None,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract attributes from @task function calls using intelligent heuristics.

    This function automatically extracts relevant information from function arguments,
    handling special cases for LLM models, Pydantic models, dataclasses, etc.
    """
    import dataclasses
    import inspect
    import json

    if ignore_args is None:
        ignore_args = set()
    if extract_fields is None:
        extract_fields = {}

    try:
        from langchain_core.language_models.chat_models import BaseChatModel
    except ImportError:
        BaseChatModel = type("BaseChatModel", (), {})

    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = type("BaseModel", (), {})  # noqa: F841

    attributes = {}

    try:
        # Get the BASE_SCOPE for attributes. TODO: should we allow users to override this?
        try:
            from trulens.otel.semconv.trace import BASE_SCOPE
        except ImportError:
            BASE_SCOPE = "trulens.task"

        sig = inspect.signature(func)

        # Merge args and kwargs to avoid duplicates
        all_kwargs = {}
        bound_args = sig.bind_partial(*args)
        all_kwargs.update(bound_args.arguments)
        all_kwargs.update(kwargs)
        bound = sig.bind(**all_kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            if name in ignore_args:
                continue

            # Skip LLM-related objects as they're not serializable
            try:
                if isinstance(value, BaseChatModel):
                    continue
            except (TypeError, NameError):
                # Handle case where types are mock objects
                pass

            # Extract only a specific field if specified
            if name in extract_fields:
                attr_path = extract_fields[name]
                for attr in attr_path.split("."):
                    value = getattr(value, attr, None)
                val = json.dumps(value, default=str, indent=2)
            else:
                # Handle different data types intelligently
                if dataclasses.is_dataclass(value):
                    val = json.dumps(
                        dataclasses.asdict(value), default=str, indent=2
                    )
                elif hasattr(value, "model_dump") or hasattr(value, "dict"):
                    # Handle Pydantic models (both v1 and v2)
                    try:
                        model_data = (
                            value.model_dump()
                            if hasattr(value, "model_dump")
                            else value.dict()
                        )
                        val = json.dumps(model_data, default=str, indent=2)
                    except (TypeError, AttributeError):
                        val = json.dumps(value, default=str, indent=2)
                else:
                    val = json.dumps(value, default=str, indent=2)

            attributes[f"{BASE_SCOPE}.{name}"] = val

        # Add return value information
        if dataclasses.is_dataclass(ret):
            ret_val = dataclasses.asdict(ret)
        elif hasattr(ret, "model_dump") or hasattr(ret, "dict"):
            # Handle Pydantic models (both v1 and v2)
            try:
                ret_val = (
                    ret.model_dump()
                    if hasattr(ret, "model_dump")
                    else ret.dict()
                )
            except (TypeError, AttributeError):
                ret_val = ret
        else:
            ret_val = ret

        attributes[f"{BASE_SCOPE}.return"] = json.dumps(
            ret_val, default=str, indent=2
        )

        attributes[f"{BASE_SCOPE}.exception"] = str(exc) if exc else ""

    except Exception as e:
        logger.exception(
            f"Exception occurred during TruLens @task instrumentation: {e}"
        )

    return attributes


TruGraph.model_rebuild()
