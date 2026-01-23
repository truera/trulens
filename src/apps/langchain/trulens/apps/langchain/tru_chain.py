"""LangChain app instrumentation."""

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
    Sequence,
)

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM

try:
    # Newer LangChain
    from langchain_core.language_models.chat_models import (
        BaseChatModel,  # type: ignore
    )
except Exception:
    try:
        # Legacy
        from langchain.chat_models.base import BaseChatModel  # type: ignore
    except Exception:
        BaseChatModel = None  # type: ignore[assignment]
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.base import RunnableSerializable

# import nest_asyncio # NOTE(piotrm): disabling for now, need more investigation
from pydantic import Field
from trulens.apps.langchain import guardrails as langchain_guardrails
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.instruments import InstrumentedMethod
from trulens.core.schema import select as select_schema
from trulens.core.session import TruSession
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.otel.semconv.trace import SpanAttributes

# Optional LangGraph support (LC agents may use Pregel under the hood)
try:
    from langgraph.pregel import Pregel  # type: ignore
except Exception:
    Pregel = None  # type: ignore[assignment]

# Optional VectorStore base for direct similarity_search instrumentation
try:
    from langchain_core.vectorstores import VectorStore  # type: ignore
except Exception:
    try:
        from langchain.vectorstores.base import VectorStore  # type: ignore
    except Exception:
        VectorStore = None  # type: ignore[assignment]

# Optional VectorStoreRetriever concrete retriever (used by as_retriever)
try:
    from langchain_core.vectorstores import VectorStoreRetriever  # type: ignore
except Exception:
    try:
        from langchain.vectorstores.base import (
            VectorStoreRetriever,  # type: ignore
        )
    except Exception:
        VectorStoreRetriever = None  # type: ignore[assignment]

# Handle langchain version compatibility for agent imports
# Be defensive: importing langchain.agents can pull in modules that may not
# exist depending on installed versions (e.g., langchain_core.memory).
try:
    # langchain <1.0
    from langchain.agents.agent import BaseMultiActionAgent  # type: ignore
    from langchain.agents.agent import BaseSingleActionAgent  # type: ignore

    try:
        from langchain.agents import (
            AgentExecutor as _AgentExecutor,  # type: ignore
        )
    except Exception:
        try:
            from langchain.agents.agent import (
                AgentExecutor as _AgentExecutor,  # type: ignore
            )
        except Exception:
            _AgentExecutor = None  # type: ignore[assignment]
except Exception:
    try:
        # langchain >=1.0
        from langchain.agents import BaseMultiActionAgent  # type: ignore
        from langchain.agents import BaseSingleActionAgent  # type: ignore

        try:
            from langchain.agents import (
                AgentExecutor as _AgentExecutor,  # type: ignore
            )
        except Exception:
            _AgentExecutor = None  # type: ignore[assignment]
    except Exception:
        # If agents are unavailable or import chains pull unsupported modules,
        # skip agent instrumentation gracefully.
        BaseMultiActionAgent = None  # type: ignore[assignment]
        BaseSingleActionAgent = None  # type: ignore[assignment]
        _AgentExecutor = None  # type: ignore[assignment]

# Try to import classic Chain for backward compatibility (e.g., LLMChain.run).
# If unavailable in the installed langchain version, gracefully degrade.
try:
    # langchain <1.0 and many 0.x versions
    from langchain.chains.base import Chain  # type: ignore
except ImportError:
    Chain = None  # type: ignore[assignment]

# Handle langchain version compatibility for serialization
try:
    # langchain <1.0
    from langchain.load.serializable import Serializable
except ImportError:
    # langchain >=1.0
    from langchain_core.load.serializable import Serializable

# Handle langchain version compatibility for memory
try:
    # Prefer 1.x community package first
    from langchain_community.chat_message_histories import (
        BaseChatMemory,  # type: ignore
    )
except Exception:
    try:
        # langchain >=0.3 sometimes exposes in langchain.memory
        from langchain.memory import BaseChatMemory  # type: ignore
    except Exception:
        try:
            # legacy path
            from langchain.memory.chat_memory import (
                BaseChatMemory,  # type: ignore
            )
        except Exception:
            BaseChatMemory = None  # type: ignore[assignment]

# Handle langchain version compatibility for prompts
try:
    # langchain <1.0
    from langchain.prompts.base import BasePromptTemplate
except ImportError:
    # langchain >=1.0
    from langchain_core.prompts import BasePromptTemplate

# Handle langchain version compatibility for retrievers
try:
    # Prefer 1.x community package first
    from langchain_community.retrievers import (
        MultiQueryRetriever,  # type: ignore
    )
except Exception:
    try:
        # 1.x sometimes keeps it under langchain.retrievers
        from langchain.retrievers import MultiQueryRetriever  # type: ignore
    except Exception:
        try:
            # legacy path
            from langchain.retrievers.multi_query import (
                MultiQueryRetriever,  # type: ignore
            )
        except Exception:
            MultiQueryRetriever = None  # type: ignore[assignment]

# Handle langchain version compatibility for schema-like imports
try:
    # Prefer 1.x first
    from langchain_core.chat_history import (
        BaseChatMessageHistory,  # type: ignore
    )
except Exception:
    try:
        from langchain_community.chat_message_histories import (  # type: ignore
            BaseChatMessageHistory,
        )
    except Exception:
        # Legacy fallback
        from langchain.schema import BaseChatMessageHistory  # type: ignore

# BaseMemory in 1.x is either in langchain_classic or langchain_core; make optional
try:
    from langchain_classic.base_memory import BaseMemory  # type: ignore
except Exception:
    try:
        from langchain_core.memory import BaseMemory  # type: ignore
    except Exception:
        BaseMemory = None  # type: ignore[assignment]

try:
    from langchain_core.documents import Document  # type: ignore
    from langchain_core.retrievers import BaseRetriever  # type: ignore
except Exception:
    # Legacy fallback
    from langchain.schema import BaseRetriever  # type: ignore
    from langchain.schema.document import Document  # type: ignore

# Handle langchain version compatibility for tools
try:
    # langchain <1.0
    from langchain.tools.base import BaseTool

    try:
        from langchain.tools import (
            StructuredTool as _StructuredTool,  # type: ignore
        )
    except Exception:
        _StructuredTool = None  # type: ignore[assignment]
except ImportError:
    # langchain >=1.0
    try:
        from langchain_core.tools import BaseTool

        try:
            from langchain_core.tools import (
                StructuredTool as _StructuredTool,  # type: ignore
            )
        except Exception:
            _StructuredTool = None  # type: ignore[assignment]
    except ImportError:
        from langchain_community.tools import BaseTool

        try:
            from langchain_community.tools import (
                StructuredTool as _StructuredTool,  # type: ignore
            )
        except Exception:
            _StructuredTool = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Handle optional MCP client import for MCP-specific instrumentation
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception as e:
    MultiServerMCPClient = None  # type: ignore[assignment]
    logger.debug(
        "langchain_mcp_adapters not available, MCP instrumentation disabled: %s",
        e,
    )

pp = PrettyPrinter()


class LangChainInstrument(core_instruments.Instrument):
    """Instrumentation for LangChain apps."""

    class Default:
        """Instrumentation specification for LangChain apps."""

        MODULES = {
            "langchain",
            "langchain_core",
            "langchain_community",
            "langchain_mcp_adapters",
            # Include langgraph to capture agent streams backed by Pregel
            "langgraph",
        }
        """Filter for module name prefix for modules to be instrumented."""

        CLASSES = (
            lambda: {
                cls
                for cls in [
                    RunnableSerializable,
                    Serializable,
                    Document,
                    Chain,
                    Pregel,  # Instrument Pregel when present
                    VectorStore,  # Instrument VectorStore similarity_search
                    _AgentExecutor,  # include AgentExecutor explicitly when available
                    BaseRetriever,
                    BaseLLM,
                    BasePromptTemplate,
                    BaseMemory,  # no methods instrumented; may be None if not available
                    BaseChatMemory,  # no methods instrumented
                    BaseChatMessageHistory,  # subclass of above
                    # langchain.agents.agent.AgentExecutor, # is langchain.chains.base.Chain
                    BaseSingleActionAgent,
                    BaseMultiActionAgent,
                    BaseLanguageModel,
                    # langchain.load.serializable.Serializable, # this seems to be work in progress over at langchain
                    # langchain.adapters.openai.ChatCompletion, # no bases
                    BaseTool,
                    langchain_guardrails.WithFeedbackFilterDocuments,
                ]
                if cls is not None
            }
        )
        """Filter for classes to be instrumented."""

        # Instrument only methods with these names and of these classes.
        @staticmethod
        def METHODS() -> List[InstrumentedMethod]:
            methods: List[InstrumentedMethod] = [
                # Mark LLM calls as GENERATION
                InstrumentedMethod(
                    "invoke",
                    BaseLanguageModel,
                    SpanAttributes.SpanType.GENERATION,
                ),
                InstrumentedMethod(
                    "ainvoke",
                    BaseLanguageModel,
                    SpanAttributes.SpanType.GENERATION,
                ),
                InstrumentedMethod(
                    "stream",
                    BaseLanguageModel,
                    SpanAttributes.SpanType.GENERATION,
                ),
                InstrumentedMethod(
                    "astream",
                    BaseLanguageModel,
                    SpanAttributes.SpanType.GENERATION,
                ),
                # Some chat models may derive from BaseChatModel separately
                *(
                    [
                        InstrumentedMethod(
                            "invoke",
                            BaseChatModel,
                            SpanAttributes.SpanType.GENERATION,
                        ),
                        InstrumentedMethod(
                            "ainvoke",
                            BaseChatModel,
                            SpanAttributes.SpanType.GENERATION,
                        ),
                        InstrumentedMethod(
                            "stream",
                            BaseChatModel,
                            SpanAttributes.SpanType.GENERATION,
                        ),
                        InstrumentedMethod(
                            "astream",
                            BaseChatModel,
                            SpanAttributes.SpanType.GENERATION,
                        ),
                    ]
                    if BaseChatModel is not None
                    else []
                ),
                # Generic runnable instrumentation (fallback when no specific span type applies)
                InstrumentedMethod(
                    "invoke", Runnable, must_be_first_wrapper=False
                ),
                InstrumentedMethod(
                    "ainvoke", Runnable, must_be_first_wrapper=False
                ),
                InstrumentedMethod("stream", Runnable),
                InstrumentedMethod("astream", Runnable),
                # Also instrument event-style streaming APIs when used
                InstrumentedMethod("stream_events", Runnable),
                InstrumentedMethod("astream_events", Runnable),
                # AgentExecutor may not be a Runnable in some versions; instrument directly.
                InstrumentedMethod(
                    "stream",
                    object
                    if "_AgentExecutor" not in globals()
                    or _AgentExecutor is None
                    else _AgentExecutor,
                ),
                InstrumentedMethod(
                    "astream",
                    object
                    if "_AgentExecutor" not in globals()
                    or _AgentExecutor is None
                    else _AgentExecutor,
                ),
                InstrumentedMethod(
                    "stream_events",
                    object
                    if "_AgentExecutor" not in globals()
                    or _AgentExecutor is None
                    else _AgentExecutor,
                ),
                InstrumentedMethod(
                    "astream_events",
                    object
                    if "_AgentExecutor" not in globals()
                    or _AgentExecutor is None
                    else _AgentExecutor,
                ),
                # Pregel methods (langgraph) frequently used by agents underneath
                InstrumentedMethod(
                    "invoke", object if Pregel is None else Pregel
                ),
                InstrumentedMethod(
                    "ainvoke", object if Pregel is None else Pregel
                ),
                InstrumentedMethod(
                    "stream", object if Pregel is None else Pregel
                ),
                InstrumentedMethod(
                    "astream", object if Pregel is None else Pregel
                ),
                InstrumentedMethod(
                    "stream_mode", object if Pregel is None else Pregel
                ),
                # VectorStore direct retrieval to emit RETRIEVAL spans
                InstrumentedMethod(
                    "similarity_search",
                    object if VectorStore is None else VectorStore,
                    *core_instruments.Instrument.Default.retrieval_span(
                        "query"
                    ),
                ),
                InstrumentedMethod(
                    "asimilarity_search",
                    object if VectorStore is None else VectorStore,
                    *core_instruments.Instrument.Default.retrieval_span(
                        "query"
                    ),
                ),
                # Properly mark retrieval spans on retrievers
                InstrumentedMethod(
                    "_get_relevant_documents",
                    BaseRetriever,
                    *core_instruments.Instrument.Default.retrieval_span(
                        "query"
                    ),
                ),
                InstrumentedMethod(
                    "get_relevant_documents",
                    BaseRetriever,
                    *core_instruments.Instrument.Default.retrieval_span(
                        "query"
                    ),
                ),
                InstrumentedMethod(
                    "aget_relevant_documents",
                    BaseRetriever,
                    *core_instruments.Instrument.Default.retrieval_span(
                        "query"
                    ),
                ),
                InstrumentedMethod(
                    "_aget_relevant_documents",
                    BaseRetriever,
                    *core_instruments.Instrument.Default.retrieval_span(
                        "query"
                    ),
                ),
                # MCP client methods - only instrument if MultiServerMCPClient is available
                *(
                    [
                        InstrumentedMethod(
                            "get_tools",
                            MultiServerMCPClient,  # Only match actual MCP client classes
                            *core_instruments.Instrument.Default.mcp_span(
                                "server_name"
                            ),
                        ),
                    ]
                    if MultiServerMCPClient is not None
                    else []
                ),
            ]

            # Optional: memory methods
            try:
                if BaseMemory is not None:  # type: ignore[name-defined]
                    methods.extend([
                        InstrumentedMethod("save_context", BaseMemory),
                        InstrumentedMethod("clear", BaseMemory),
                    ])
            except NameError:
                pass

            # Optional: classic Chain methods
            try:
                if Chain is not None:  # type: ignore[name-defined]
                    methods.extend([
                        InstrumentedMethod("run", Chain),
                        InstrumentedMethod("arun", Chain),
                        InstrumentedMethod("_call", Chain),
                        InstrumentedMethod("__call__", Chain),
                        InstrumentedMethod("_acall", Chain),
                        InstrumentedMethod("acall", Chain),
                    ])
            except NameError:
                pass

            # Optional: agent plan methods
            try:
                if BaseSingleActionAgent is not None:  # type: ignore[name-defined]
                    methods.append(
                        InstrumentedMethod("plan", BaseSingleActionAgent)
                    )
                    methods.append(
                        InstrumentedMethod("aplan", BaseSingleActionAgent)
                    )
            except NameError:
                pass

            try:
                if BaseMultiActionAgent is not None:  # type: ignore[name-defined]
                    methods.append(
                        InstrumentedMethod("plan", BaseMultiActionAgent)
                    )
                    methods.append(
                        InstrumentedMethod("aplan", BaseMultiActionAgent)
                    )
            except NameError:
                pass

            # Tools
            try:
                methods.extend([
                    InstrumentedMethod(
                        "_arun",
                        BaseTool,
                        # Default tools are TOOL; MCP adapters will override to MCP via their own wrappers
                        SpanAttributes.SpanType.TOOL,
                        lambda ret, exception, *args, **kwargs: {
                            SpanAttributes.MCP.TOOL_NAME: getattr(
                                args[0], "name", "unknown"
                            )
                            if args
                            else "unknown",
                            SpanAttributes.MCP.INPUT_ARGUMENTS: str(args[1:])
                            if len(args) > 1
                            else str(kwargs),
                            SpanAttributes.MCP.OUTPUT_CONTENT: str(ret)
                            if ret is not None
                            else "",
                            SpanAttributes.MCP.OUTPUT_IS_ERROR: exception
                            is not None,
                        },
                    ),
                    InstrumentedMethod(
                        "_run",
                        BaseTool,
                        # Default tools are TOOL; MCP adapters will override to MCP via their own wrappers
                        SpanAttributes.SpanType.TOOL,
                        lambda ret, exception, *args, **kwargs: {
                            SpanAttributes.MCP.TOOL_NAME: getattr(
                                args[0], "name", "unknown"
                            )
                            if args
                            else "unknown",
                            SpanAttributes.MCP.INPUT_ARGUMENTS: str(args[1:])
                            if len(args) > 1
                            else str(kwargs),
                            SpanAttributes.MCP.OUTPUT_CONTENT: str(ret)
                            if ret is not None
                            else "",
                            SpanAttributes.MCP.OUTPUT_IS_ERROR: exception
                            is not None,
                        },
                    ),
                ])
            except NameError:
                pass

            return methods

        """Methods to be instrumented.

        Key is method name and value is filter for objects that need those
        methods instrumented"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LangChainInstrument.Default.MODULES,
            include_classes=LangChainInstrument.Default.CLASSES(),
            include_methods=LangChainInstrument.Default.METHODS(),
            *args,
            **kwargs,
        )


class TruChain(core_app.App):
    """Recorder for _LangChain_ applications.

    This recorder is designed for LangChain apps, providing a way to instrument,
    log, and evaluate their behavior.

    Example: "Creating a LangChain RAG application"

        Consider an example LangChain RAG application. For the complete code
        example, see [LangChain
        Quickstart](https://www.trulens.org/getting_started/quickstarts/langchain_quickstart/).

        ```python
        from langchain import hub
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        retriever = vectorstore.as_retriever()

        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        ```

    Feedback functions can utilize the specific context produced by the
    application's retriever. This is achieved using the `select_context` method,
    which then can be used by a feedback selector, such as `on(context)`.

    Example: "Defining a feedback function"

        ```python
        from trulens.providers.openai import OpenAI
        from trulens.core import Feedback
        import numpy as np

        # Select context to be used in feedback.
        from trulens.apps.langchain import TruChain
        context = TruChain.select_context(rag_chain)


        # Use feedback
        f_context_relevance = (
            Feedback(provider.context_relevance_with_context_reasons)
            .on_input()
            .on(context)  # Refers to context defined from `select_context`
            .aggregate(np.mean)
        )
        ```

    The application can be wrapped in a `TruChain` recorder to provide logging
    and evaluation upon the application's use.

    Example: "Using the `TruChain` recorder"

        ```python
        from trulens.apps.langchain import TruChain

        # Wrap application
        tru_recorder = TruChain(
            chain,
            app_name="ChatApplication",
            app_version="chain_v1",
            feedbacks=[f_context_relevance]
        )

        # Record application runs
        with tru_recorder as recording:
            chain("What is langchain?")
        ```

    Further information about LangChain apps can be found on the [LangChain
    Documentation](https://python.langchain.com/docs/) page.

    Args:
        app: A LangChain application.

        **kwargs: Additional arguments to pass to [App][trulens.core.app.App]
            and [AppDefinition][trulens.core.schema.app.AppDefinition].
    """

    app: Runnable
    """The langchain app to be instrumented."""

    # TODEP
    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)
    """The root callable of the wrapped app."""

    # Normally pydantic does not like positional args but chain here is
    # important enough to make an exception.
    def __init__(
        self,
        app: Runnable,
        main_method: Optional[Callable] = None,
        **kwargs: Dict[str, Any],
    ):
        # TruChain specific:
        kwargs["app"] = app
        # Create `TruSession` if not already created.
        if "connector" in kwargs:
            TruSession(connector=kwargs["connector"])
        else:
            TruSession()

        # Ensure class-level instrumentation for common components under OTEL
        try:
            from trulens.core.experimental import Feature as _Feature
            from trulens.core.otel.instrument import (
                instrument_method as _otel_instrument_method,
            )

            otel_enabled = TruSession().experimental_feature(
                _Feature.OTEL_TRACING
            )
        except Exception:
            otel_enabled = False

        if otel_enabled:
            # Idempotent guard
            if not hasattr(TruChain, "_otel_langchain_class_instrumented"):
                try:
                    # Generation spans for LLMs
                    try:
                        _otel_instrument_method(
                            cls=BaseLanguageModel,
                            method_name="invoke",
                            span_type=SpanAttributes.SpanType.GENERATION,
                        )
                        _otel_instrument_method(
                            cls=BaseLanguageModel,
                            method_name="ainvoke",
                            span_type=SpanAttributes.SpanType.GENERATION,
                        )
                        _otel_instrument_method(
                            cls=BaseLanguageModel,
                            method_name="stream",
                            span_type=SpanAttributes.SpanType.GENERATION,
                        )
                        _otel_instrument_method(
                            cls=BaseLanguageModel,
                            method_name="astream",
                            span_type=SpanAttributes.SpanType.GENERATION,
                        )
                        # Chat model base (some LC versions)
                        try:
                            from langchain_core.language_models.chat_models import (
                                BaseChatModel as _BaseChatModel,  # type: ignore
                            )
                        except Exception:
                            try:
                                from langchain.chat_models.base import (
                                    BaseChatModel as _BaseChatModel,  # type: ignore
                                )
                            except Exception:
                                _BaseChatModel = None  # type: ignore[assignment]
                        if _BaseChatModel is not None:
                            for _m in [
                                "invoke",
                                "ainvoke",
                                "stream",
                                "astream",
                            ]:
                                try:
                                    _otel_instrument_method(
                                        cls=_BaseChatModel,
                                        method_name=_m,
                                        span_type=SpanAttributes.SpanType.GENERATION,
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Retrieval spans for retrievers
                    try:
                        _otel_instrument_method(
                            cls=BaseRetriever,
                            method_name="get_relevant_documents",
                            span_type=SpanAttributes.SpanType.RETRIEVAL,
                            attributes=core_instruments.Instrument.Default.retrieval_span(
                                "query"
                            )[1],
                        )
                        _otel_instrument_method(
                            cls=BaseRetriever,
                            method_name="aget_relevant_documents",
                            span_type=SpanAttributes.SpanType.RETRIEVAL,
                            attributes=core_instruments.Instrument.Default.retrieval_span(
                                "query"
                            )[1],
                        )
                        # Also instrument the concrete VectorStoreRetriever
                        if VectorStoreRetriever is not None:
                            for _m in [
                                "get_relevant_documents",
                                "aget_relevant_documents",
                                "_get_relevant_documents",
                                "_aget_relevant_documents",
                            ]:
                                try:
                                    _otel_instrument_method(
                                        cls=VectorStoreRetriever,
                                        method_name=_m,
                                        span_type=SpanAttributes.SpanType.RETRIEVAL,
                                        attributes=core_instruments.Instrument.Default.retrieval_span(
                                            "query"
                                        )[1],
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Retrieval spans for VectorStore direct usage
                    try:
                        if VectorStore is not None:
                            _retrieval_attr = core_instruments.Instrument.Default.retrieval_span(
                                "query"
                            )[1]
                            _vs_methods = [
                                "similarity_search",
                                "asimilarity_search",
                                "similarity_search_with_relevance_scores",
                                "asimilarity_search_with_relevance_scores",
                                "max_marginal_relevance_search",
                                "amax_marginal_relevance_search",
                            ]
                            # Instrument base
                            for _m in _vs_methods:
                                try:
                                    _otel_instrument_method(
                                        cls=VectorStore,
                                        method_name=_m,
                                        span_type=SpanAttributes.SpanType.RETRIEVAL,
                                        attributes=_retrieval_attr,
                                        must_be_first_wrapper=True,
                                    )
                                except Exception:
                                    pass
                            # Instrument currently loaded subclasses as well
                            try:

                                def _instrument_subclasses(base):
                                    for sub in list(
                                        getattr(
                                            base, "__subclasses__", lambda: []
                                        )()
                                    ):
                                        for _m in _vs_methods:
                                            if hasattr(sub, _m):
                                                try:
                                                    _otel_instrument_method(
                                                        cls=sub,
                                                        method_name=_m,
                                                        span_type=SpanAttributes.SpanType.RETRIEVAL,
                                                        attributes=_retrieval_attr,
                                                        must_be_first_wrapper=True,
                                                    )
                                                except Exception:
                                                    pass
                                        # Recurse
                                        _instrument_subclasses(sub)

                                _instrument_subclasses(VectorStore)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Do not force TOOL overrides at the OTEL layer; rely on Default.METHODS.

                    # Finally, ensure LangGraph class-level instrumentation (graph nodes/tasks).
                    # Doing this last ensures our LLM/retriever/tool span types are preserved.
                    try:
                        from trulens.apps.langgraph.tru_graph import (
                            TruGraph,  # type: ignore
                        )

                        TruGraph._ensure_instrumentation()  # idempotent
                    except Exception:
                        pass

                    # Mark as done
                    setattr(
                        TruChain, "_otel_langchain_class_instrumented", True
                    )
                except Exception:
                    pass

        if main_method is not None:
            kwargs["main_method"] = main_method
        kwargs["root_class"] = pyschema_utils.Class.of_object(app)
        kwargs["instrument"] = LangChainInstrument(app=self)

        super().__init__(**kwargs)

    @classmethod
    def select_context(cls, app: Optional[Chain] = None) -> serial_utils.Lens:
        """Get the path to the context in the query output."""

        if app is None:
            raise ValueError(
                "langchain app/chain is required to determine context for langchain apps. "
                "Pass it in as the `app` argument"
            )

        retrievers = []

        app_json = json_utils.jsonify(app)
        for lens in serial_utils.all_queries(app_json):
            try:
                comp = lens.get_sole_item(app)
                if isinstance(comp, BaseRetriever):
                    retrievers.append((lens, comp))

            except Exception:
                pass

        if len(retrievers) == 0:
            raise ValueError("Cannot find any `BaseRetriever` in app.")

        if len(retrievers) > 1:
            if isinstance(retrievers[0][1], MultiQueryRetriever):
                pass
            else:
                raise ValueError(
                    "Found more than one `BaseRetriever` in app:\n\t"
                    + (
                        "\n\t".join(
                            map(
                                lambda lr: f"{type(lr[1])} at {lr[0]}",
                                retrievers,
                            )
                        )
                    )
                )

        retriever = select_schema.Select.RecordCalls + retrievers[0][0]
        if hasattr(retriever, "invoke"):
            return retriever.invoke.rets[:].page_content

        if hasattr(retriever, "_get_relevant_documents"):
            return retriever._get_relevant_documents.rets[:].page_content

        raise RuntimeError("Could not find a retriever component in app.")

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:  # might have to relax to JSON output
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """

        # Prefer LCEL-style messages input if present
        try:
            # Case 1: messages passed directly
            if "messages" in bindings.arguments:
                msgs = bindings.arguments["messages"]
                if isinstance(msgs, Sequence) and len(msgs) > 0:
                    last = msgs[-1]
                    if hasattr(last, "content"):
                        return getattr(last, "content")
                    if isinstance(last, Dict) and isinstance(
                        last.get("content"), str
                    ):
                        return last.get("content")
            # Case 2: messages nested inside input dict
            if "input" in bindings.arguments and isinstance(
                bindings.arguments["input"], Dict
            ):
                msgs = bindings.arguments["input"].get("messages")
                if isinstance(msgs, Sequence) and len(msgs) > 0:
                    last = msgs[-1]
                    if hasattr(last, "content"):
                        return getattr(last, "content")
                    if isinstance(last, Dict) and isinstance(
                        last.get("content"), str
                    ):
                        return last.get("content")
        except Exception:
            pass

        if "input" in bindings.arguments:
            temp = bindings.arguments["input"]
            if isinstance(temp, str):
                return temp

            if isinstance(temp, dict):
                vals = list(temp.values())
            elif isinstance(temp, list):
                vals = temp
            else:
                vals = [temp]

            if len(vals) == 0:
                return None

            if len(vals) == 1:
                return vals[0]

            if len(vals) > 1:
                return vals[0]

        if (
            "inputs" in bindings.arguments
            and python_utils.safe_hasattr(self.app, "input_keys")
            and python_utils.safe_hasattr(self.app, "prep_inputs")
        ):
            # langchain specific:
            ins = self.app.prep_inputs(bindings.arguments["inputs"])

            if len(self.app.input_keys) == 0:
                logger.warning(
                    "langchain app has no `input_keys`. `main_input` might not be detected."
                )
                return super().main_input(func, sig, bindings)

            return ins[self.app.input_keys[0]]

        return core_app.App.main_input(self, func, sig, bindings)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> str:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        # Preserve simple JSON-friendly returns (strings, numbers, etc.) verbatim.
        if isinstance(ret, serial_utils.JSON_BASES):
            return str(ret)

        # Prefer LCEL results shape: dict with "messages" or list of Message-like objects
        try:
            # Case 1: result dict with "messages"
            if (
                isinstance(ret, Dict)
                and "messages" in ret
                and isinstance(ret["messages"], Sequence)
            ):
                messages = ret["messages"]
                # Pick last message with textual content
                for last in reversed(messages):
                    if hasattr(last, "content") and getattr(last, "content"):
                        return getattr(last, "content")
                    if isinstance(last, Dict) and isinstance(
                        last.get("content"), str
                    ):
                        if last.get("content"):
                            return last.get("content")
            # Case 2: result is a sequence of Message-like items
            if isinstance(ret, Sequence) and len(ret) > 0:
                for last in reversed(ret):
                    if hasattr(last, "content") and getattr(last, "content"):
                        return getattr(last, "content")
                    if isinstance(last, Dict) and isinstance(
                        last.get("content"), str
                    ):
                        if last.get("content"):
                            return last.get("content")
        except Exception:
            pass

        if isinstance(ret, (AIMessage, AIMessageChunk)):
            return ret.content

        if isinstance(ret, Sequence) and all(
            isinstance(x, Dict) and "content" in x for x in ret
        ):
            # Streaming outputs for some internal methods are lists of dicts
            # with each having "content".
            return "".join(x["content"] for x in ret)

        # Handle event-style streaming outputs (e.g., Runnable.stream_events/astream_events)
        if isinstance(ret, Sequence) and all(isinstance(x, Dict) for x in ret):

            def _extract_event_content(e: Dict) -> str:
                # Direct content
                if "content" in e and isinstance(e["content"], str):
                    return e["content"]
                # Values-mode dicts: often {'messages': [...]} from stream(stream_mode="values")
                msgs = e.get("messages")
                if isinstance(msgs, list) and len(msgs) > 0:
                    last_msg = msgs[-1]
                    if hasattr(last_msg, "content"):
                        return getattr(last_msg, "content") or ""
                    if isinstance(last_msg, Dict) and isinstance(
                        last_msg.get("content"), str
                    ):
                        return last_msg.get("content", "") or ""
                # Nested data payloads used by LangChain event streams
                data = e.get("data")
                if isinstance(data, Dict):
                    # Agent final outputs often appear under 'output' or 'return_values'
                    if isinstance(data.get("output"), str):
                        return data.get("output") or ""
                    rv = data.get("return_values")
                    if isinstance(rv, Dict):
                        # Common shape: {'output': '...'}
                        if isinstance(rv.get("output"), str):
                            return rv.get("output") or ""
                        # Sometimes nested under another key (first string wins)
                        for v in rv.values():
                            if isinstance(v, str):
                                return v
                    # Messages list with AIMessage-like objects
                    messages = data.get("messages")
                    if isinstance(messages, list) and len(messages) > 0:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "content"):
                            return getattr(last_msg, "content") or ""
                        if isinstance(last_msg, Dict) and isinstance(
                            last_msg.get("content"), str
                        ):
                            return last_msg.get("content", "") or ""
                    chunk = data.get("chunk")
                    # chunk may be an AIMessageChunk-like object or a dict
                    if hasattr(chunk, "content"):
                        return getattr(chunk, "content") or ""
                    if isinstance(chunk, Dict) and isinstance(
                        chunk.get("content"), str
                    ):
                        return chunk.get("content", "") or ""
                    # Some events carry output/result objects
                    output = data.get("output") or data.get("result")
                    if hasattr(output, "content"):
                        return getattr(output, "content") or ""
                    if isinstance(output, Dict) and isinstance(
                        output.get("content"), str
                    ):
                        return output.get("content", "") or ""
                    # Some agent events carry 'final_output' or similar
                    for key in ("final_output", "final_answer"):
                        val = data.get(key)
                        if isinstance(val, str):
                            return val
                return ""

            parts = [_extract_event_content(x) for x in ret]
            if any(parts):
                return "".join(parts)

        if isinstance(ret, Sequence) and all(
            isinstance(x, (AIMessage, AIMessageChunk)) for x in ret
        ):
            # Streaming outputs for some internal methods are lists of dicts
            # with each having "content".
            return "".join(x.content for x in ret)

        if isinstance(ret, Sequence) and all(isinstance(x, str) for x in ret):
            # Streaming outputs of main stream methods like Runnable.stream are
            # sometimes bundled by us into a sequence of strings that includes
            # intermediary tool messages. Prefer the last non-empty string which
            # is typically the final assistant output.
            for s in reversed(ret):
                if isinstance(s, str) and s.strip():
                    return s
            return "".join(ret)

        if isinstance(ret, Dict) and python_utils.safe_hasattr(
            self.app, "output_keys"
        ):
            # langchain specific:
            if len(self.app.output_keys) == 0:
                logger.warning(
                    "langchain app has no `output_keys`. `main_output` might not be detected."
                )
                return super().main_output(func, sig, bindings, ret)

            if self.app.output_keys[0] in ret:
                return ret[self.app.output_keys[0]]

        return core_app.App.main_output(self, func, sig, bindings, ret)

    def main_call(self, human: str):
        # If available, a single text to a single text invocation of this app.

        if python_utils.safe_hasattr(self.app, "output_keys"):
            out_key = self.app.output_keys[0]
            return self.app(human)[out_key]
        else:
            logger.warning("Unsure what the main output string may be.")
            return str(self.app(human))

    async def main_acall(self, human: str):
        # If available, a single text to a single text invocation of this app.

        out = await self._acall(human)

        if python_utils.safe_hasattr(self.app, "output_keys"):
            out_key = self.app.output_keys[0]
            return out[out_key]
        else:
            logger.warning("Unsure what the main output string may be.")
            return str(out)

    # NOTE: Input signature compatible with langchain.chains.base.Chain.acall
    # TOREMOVE
    async def acall_with_record(self, *args, **kwargs) -> None:
        """
        DEPRECATED: Run the chain acall method and also return a record metadata object.
        """

        self._throw_dep_message(method="acall", is_async=True, with_record=True)

    # NOTE: Input signature compatible with langchain.chains.base.Chain.__call__
    # TOREMOVE
    def call_with_record(self, *args, **kwargs) -> None:
        """
        DEPRECATED: Run the chain call method and also return a record metadata object.
        """

        self._throw_dep_message(
            method="__call__", is_async=False, with_record=True
        )

    # TOREMOVE
    # Chain requirement
    def _call(self, *args, **kwargs) -> None:
        self._throw_dep_message(
            method="_call", is_async=False, with_record=False
        )

    # TOREMOVE
    # Optional Chain requirement
    async def _acall(self, *args, **kwargs) -> None:
        self._throw_dep_message(
            method="_acall", is_async=True, with_record=False
        )


TruChain.model_rebuild()
