"""LangGraph app instrumentation."""

from inspect import BoundArguments
from inspect import Signature
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
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
from trulens.otel.semconv.constants import TRULENS_INSTRUMENT_WRAPPER_FLAG
from trulens.otel.semconv.trace import SpanAttributes

from langgraph.func import TaskFunction
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command

logger = logging.getLogger(__name__)


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
            # Note: TaskFunction is instrumented at class-level during initialization
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
    - `TaskFunction.__call__` method - instrumented at class-level during import
    - `StateGraph` objects (uncompiled graphs) for logging/debugging purposes

    **Class-Level Instrumentation**: Both `@task` functions (TaskFunction) and
    `Pregel` graph methods are instrumented at the class level when TruGraph is imported.
    This ensures all function calls are captured regardless of where the instances
    are embedded in the object hierarchy (e.g., inside custom classes).

    **Benefits of Class-Level Approach**:
    - **Guaranteed Coverage**: All TaskFunction and Pregel method calls are captured
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
        """Set up class-level instrumentation for TaskFunction.__call__"""

        if not is_otel_tracing_enabled():
            logger.debug(
                "OTEL not enabled, skipping TaskFunction class-level instrumentation"
            )
            return

        try:
            from trulens.core.otel.instrument import instrument_method

            # Check if TaskFunction.__call__ is already instrumented
            if hasattr(TaskFunction.__call__, TRULENS_INSTRUMENT_WRAPPER_FLAG):
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

                    if hasattr(task_function_instance, "func") and hasattr(
                        task_function_instance.func, "__name__"
                    ):
                        attributes[SpanAttributes.GRAPH_TASK.TASK_NAME] = (
                            task_function_instance.func.__name__
                        )

                    # Serialize the task input arguments
                    try:
                        import inspect
                        import json

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
                "Successfully applied class-level instrumentation to TaskFunction.__call__"
            )

        except Exception as e:
            logger.warning(
                f"Failed to apply class-level TaskFunction instrumentation: {e}"
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
            from trulens.core.otel.instrument import instrument_method

            if hasattr(Pregel, "invoke") and hasattr(
                getattr(Pregel, "invoke"), TRULENS_INSTRUMENT_WRAPPER_FLAG
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
                        attributes[SpanAttributes.GRAPH_NODE.INPUT_STATE] = str(
                            input_data
                        )
                    else:
                        attributes[SpanAttributes.GRAPH_NODE.INPUT_STATE] = str(
                            input_data
                        )

                for k, v in kwargs.items():
                    if k in ["input", "state", "data"]:
                        attributes[SpanAttributes.GRAPH_NODE.INPUT_STATE] = str(
                            v
                        )
                        break

                if ret is not None and not exception:
                    attributes[SpanAttributes.GRAPH_NODE.OUTPUT_STATE] = str(
                        ret
                    )

                if exception:
                    attributes[SpanAttributes.GRAPH_NODE.ERROR] = str(exception)

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
                    span_type=SpanAttributes.SpanType.GRAPH_NODE,
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

    @classmethod
    def _ensure_instrumentation(cls):
        """Ensure one-time initialization of instrumentation."""
        if not cls._is_instrumented:
            cls._setup_task_function_instrumentation()
            cls._setup_pregel_instrumentation()
            cls._is_instrumented = True

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
            return app.compile()  # type: ignore

        if isinstance(app, Pregel):
            return app

        return app

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
