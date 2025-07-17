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

    try:
        import os

        import wrapt

        def _task_wrapper(wrapped, instance, args, kwargs):
            """Wrapper for langgraph.func.task that applies TruLens instrumentation."""
            otel_enabled = os.environ.get("TRULENS_OTEL_TRACING", "0") == "1"

            logger.debug(f"Task wrapper called, OTEL enabled: {otel_enabled}")

            if not otel_enabled:
                logger.debug("OTEL not enabled, skipping instrumentation")
                return wrapped(*args, **kwargs)

            try:
                from trulens.core.otel.instrument import instrument

                def task_attributes(
                    ret, exception, *inner_args, **inner_kwargs
                ):
                    logger.debug(
                        f"Task attributes function called with args: {len(inner_args)}, kwargs: {len(inner_kwargs)}"
                    )
                    attributes = {}
                    try:
                        import inspect
                        import json

                        task_func = inner_args[0] if inner_args else None
                        if not task_func or not callable(task_func):
                            return attributes

                        if len(inner_args) > 1:
                            sig = inspect.signature(task_func)
                            bound_args = sig.bind_partial(
                                *inner_args[1:], **inner_kwargs
                            ).arguments
                            all_kwargs = {**inner_kwargs, **bound_args}

                            for name, value in all_kwargs.items():
                                if hasattr(value, "__module__") and (
                                    "llm" in str(type(value)).lower()
                                    or "pool" in str(type(value)).lower()
                                ):
                                    continue

                                try:
                                    if hasattr(value, "__dict__") and hasattr(
                                        value, "__dataclass_fields__"
                                    ):
                                        import dataclasses

                                        val = json.dumps(
                                            dataclasses.asdict(value),
                                            default=str,
                                            indent=2,
                                        )
                                    elif hasattr(value, "model_dump"):
                                        val = json.dumps(
                                            value.model_dump(),
                                            default=str,
                                            indent=2,
                                        )
                                    elif hasattr(value, "dict") and callable(
                                        getattr(value, "dict")
                                    ):
                                        val = json.dumps(
                                            value.dict(), default=str, indent=2
                                        )
                                    else:
                                        val = json.dumps(
                                            value, default=str, indent=2
                                        )

                                    attributes[
                                        f"trulens.langgraph_task.input_state.{name}"
                                    ] = val
                                except Exception:
                                    attributes[
                                        f"trulens.langgraph_task.input_state.{name}"
                                    ] = str(value)

                            if ret is not None and not exception:
                                try:
                                    if hasattr(ret, "__dict__") and hasattr(
                                        ret, "__dataclass_fields__"
                                    ):
                                        import dataclasses

                                        ret_val = dataclasses.asdict(ret)
                                    elif hasattr(ret, "model_dump"):
                                        ret_val = ret.model_dump()
                                    elif hasattr(ret, "dict") and callable(
                                        getattr(ret, "dict")
                                    ):
                                        ret_val = ret.dict()
                                    else:
                                        ret_val = ret
                                    attributes[
                                        "trulens.langgraph_task.output_state"
                                    ] = json.dumps(
                                        ret_val, default=str, indent=2
                                    )
                                except Exception:
                                    attributes[
                                        "trulens.langgraph_task.output_state"
                                    ] = str(ret)

                            if exception:
                                attributes["trulens.langgraph_task.error"] = (
                                    str(exception)
                                )

                    except Exception as e:
                        logger.warning(
                            f"Error in task attributes extraction: {e}"
                        )

                    return attributes

                if args and callable(args[0]):
                    func = args[0]
                    logger.info(
                        f"Auto-instrumenting @task function: {func.__name__}"
                    )

                    instrumented_func = instrument(
                        span_type=SpanAttributes.SpanType.LANGGRAPH_TASK,
                        attributes=task_attributes,
                    )(func)

                    return wrapped(instrumented_func, **kwargs)
                else:
                    original_decorator = wrapped(*args, **kwargs)

                    def new_decorator(func):
                        logger.info(
                            f"Auto-instrumenting @task function: {func.__name__}"
                        )

                        instrumented_func = instrument(
                            span_type=SpanAttributes.SpanType.LANGGRAPH_TASK,
                            attributes=task_attributes,
                        )(func)

                        return original_decorator(instrumented_func)

                    return new_decorator

            except Exception as e:
                logger.warning(f"Error in task wrapper: {e}")
                return wrapped(*args, **kwargs)

        def _entrypoint_wrapper(wrapped, instance, args, kwargs):
            """Wrapper for langgraph.func.entrypoint that applies TruLens instrumentation."""
            otel_enabled = os.environ.get("TRULENS_OTEL_TRACING", "0") == "1"

            if not otel_enabled:
                return wrapped(*args, **kwargs)

            try:
                from trulens.core.otel.instrument import instrument

                original_result = wrapped(*args, **kwargs)

                if hasattr(original_result, "__call__") and not hasattr(
                    original_result, "_trulens_instrumented"
                ):
                    logger.debug(
                        f"Auto-instrumenting @entrypoint: {original_result}"
                    )

                    instrumented_call = instrument(
                        span_type=SpanAttributes.SpanType.LANGGRAPH_ENTRYPOINT,
                        attributes=lambda ret, exception, *args, **kwargs: {
                            "trulens.langgraph_entrypoint.input_data": str(
                                args[1] if len(args) > 1 else ""
                            ),
                            "trulens.langgraph_entrypoint.output_data": str(ret)
                            if ret is not None and not exception
                            else "",
                            "trulens.langgraph_entrypoint.error": str(exception)
                            if exception
                            else "",
                        },
                    )(original_result.__call__)

                    original_result.__call__ = instrumented_call
                    original_result._trulens_instrumented = True

                return original_result

            except Exception as e:
                logger.warning(f"Error in entrypoint wrapper: {e}")
                return wrapped(*args, **kwargs)

        wrapt.wrap_function_wrapper("langgraph.func", "task", _task_wrapper)
        wrapt.wrap_function_wrapper(
            "langgraph.func", "entrypoint", _entrypoint_wrapper
        )

        logger.info(
            "Applied wrapt wrappers to @task and @entrypoint at import time"
        )

    except ImportError:
        logger.warning(
            "wrapt not available, cannot wrap @task and @entrypoint decorators"
        )
    except Exception as e:
        logger.warning(f"Error applying wrapt wrappers at import time: {e}")

except ImportError:
    task = None
    TaskFunction = type("TaskFunction", (), {})
    entrypoint = type("entrypoint", (), {})
    LANGGRAPH_FUNC_AVAILABLE = False


def _langgraph_graph_span() -> Dict[str, Any]:
    """Create span configuration for LangGraph graph operations."""

    def _attributes(ret, exception, *args, **kwargs) -> Dict[str, Any]:
        attributes = {}

        if args and len(args) > 1:
            input_data = args[1]
            if isinstance(input_data, dict):
                attributes[SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE] = str(
                    input_data
                )
            else:
                attributes[SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE] = str(
                    input_data
                )

        for k, v in kwargs.items():
            if k in ["input", "state", "data"]:
                attributes[SpanAttributes.LANGGRAPH_GRAPH.INPUT_STATE] = str(v)
                break

        if ret is not None and not exception:
            attributes[SpanAttributes.LANGGRAPH_GRAPH.OUTPUT_STATE] = str(ret)

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

        if args and len(args) > 1:
            input_data = args[1] if len(args) > 1 else args[0]
            attributes[SpanAttributes.LANGGRAPH_TASK.INPUT_STATE] = str(
                input_data
            )

        for k, v in kwargs.items():
            if k in ["state", "data", "context"]:
                attributes[SpanAttributes.LANGGRAPH_TASK.INPUT_STATE] = str(v)
                break

        if ret is not None and not exception:
            attributes[SpanAttributes.LANGGRAPH_TASK.OUTPUT_STATE] = str(ret)

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

        if args and len(args) > 1:
            input_data = args[1]
            attributes[SpanAttributes.LANGGRAPH_ENTRYPOINT.INPUT_DATA] = str(
                input_data
            )

        for k, v in kwargs.items():
            if k in ["input", "data", "query"]:
                attributes[SpanAttributes.LANGGRAPH_ENTRYPOINT.INPUT_DATA] = (
                    str(v)
                )
                break

        if ret is not None and not exception:
            attributes[SpanAttributes.LANGGRAPH_ENTRYPOINT.OUTPUT_DATA] = str(
                ret
            )

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
                entrypoint,
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

        if LANGGRAPH_AVAILABLE:
            METHODS.extend([
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

        # After initialization, help the instrumentation system find nested Pregel objects
        # This does NOT interfere with @task/@entrypoint instrumentation
        self._instrument_nested_pregel_objects()

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

        # For custom classes, only detect components for logging/debugging
        # Do NOT modify the app to avoid interfering with existing @task/@entrypoint instrumentation
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
        It does NOT interfere with existing @task/@entrypoint instrumentation.

        Args:
            app: The application object to inspect

        Returns:
            List of (attribute_name, component) tuples for found LangGraph components
        """
        components = []

        if not hasattr(app, "__dict__"):
            return components

        # Only check for basic LangGraph objects - Pregel and StateGraph
        # @task and @entrypoint instrumentation is handled separately via wrapt
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

    def _instrument_nested_pregel_objects(self) -> None:
        """
        Help the instrumentation system find and instrument nested Pregel objects.

        This method specifically targets Pregel objects (compiled LangGraph workflows)
        that might be nested within custom classes. It does NOT interfere with
        @task/@entrypoint instrumentation which is handled separately via wrapt.
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
        self, obj: Any, path: Optional[List[str]] = None
    ) -> List[Tuple[List[str], Any]]:
        """
        Recursively find Pregel objects nested within the given object.

        Args:
            obj: Object to search within
            path: Current path to this object

        Returns:
            List of (path_parts, pregel_object) tuples
        """
        if path is None:
            path = []

        pregels = []

        # If this object itself is a Pregel, add it (but skip if it's the root app)
        if (
            isinstance(obj, Pregel) and path
        ):  # path check ensures we skip root app
            pregels.append((path, obj))
            return pregels  # Don't recurse into Pregel objects

        # Look for Pregel objects in attributes
        if hasattr(obj, "__dict__"):
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue

                try:
                    attr_value = getattr(obj, attr_name)

                    if isinstance(attr_value, Pregel):
                        pregels.append((path + [attr_name], attr_value))
                    elif hasattr(attr_value, "__dict__") and not callable(
                        attr_value
                    ):
                        # Recursively search one level deep
                        nested_pregels = self._find_nested_pregel_objects(
                            attr_value, path + [attr_name]
                        )
                        pregels.extend(nested_pregels)

                except Exception:
                    continue

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

    def debug_instrumentation(self) -> Dict[str, Any]:
        """Debug method to check instrumentation status."""
        import os

        debug_info = {
            "otel_enabled": os.environ.get("TRULENS_OTEL_TRACING", "0") == "1",
            "environment_variable": os.environ.get(
                "TRULENS_OTEL_TRACING", "not_set"
            ),
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "langgraph_func_available": LANGGRAPH_FUNC_AVAILABLE,
            "detected_components": self._detect_langgraph_components(self.app),
        }

        logger.info(f"Instrumentation debug info: {debug_info}")
        return debug_info


TruGraph.model_rebuild()
