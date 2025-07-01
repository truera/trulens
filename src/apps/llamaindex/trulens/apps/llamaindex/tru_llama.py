"""LlamaIndex instrumentation."""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    TypeVar,
    Union,
)

import llama_index
from pydantic import Field
from trulens.apps.langchain import tru_chain as mod_tru_chain
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core._utils.pycompat import EmptyType  # import style exception
from trulens.core._utils.pycompat import (
    getmembers_static,  # import style exception
)
from trulens.core.instruments import InstrumentedMethod

# TODO: Do we need to depend on this?
from trulens.core.session import TruSession
from trulens.core.utils import imports as import_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.experimental.otel_tracing.core.span import Attributes
from trulens.otel.semconv.trace import SpanAttributes

T = TypeVar("T")

logger = logging.getLogger(__name__)

version = import_utils.get_package_version("llama_index")

# If llama index is not installed, will get a dummy for llama_index. In that
# case or if it is installed and sufficiently new version, continue with
# this set of imports. We don't want to partially run the new imports and
# fail midway to continue with the legacy imports.

legacy = (
    version is None
    or isinstance(llama_index, import_utils.Dummy)
    or version < import_utils.parse_version("0.10.0")
)

if not legacy:
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.base.base_query_engine import QueryEngineComponent
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.base.llms.base import BaseLLM
    from llama_index.core.base.llms.types import LLMMetadata
    from llama_index.core.base.response.schema import AsyncStreamingResponse
    from llama_index.core.base.response.schema import Response
    from llama_index.core.base.response.schema import StreamingResponse
    from llama_index.core.chat_engine.types import AgentChatResponse
    from llama_index.core.chat_engine.types import BaseChatEngine
    from llama_index.core.chat_engine.types import StreamingAgentChatResponse
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.indices.prompt_helper import PromptHelper
    from llama_index.core.memory.types import BaseMemory
    from llama_index.core.node_parser.interface import NodeParser
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.question_gen.types import BaseQuestionGenerator
    from llama_index.core.response_synthesizers import BaseSynthesizer
    from llama_index.core.response_synthesizers import Refine
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import BaseComponent
    from llama_index.core.schema import NodeWithScore
    from llama_index.core.schema import QueryBundle
    from llama_index.core.service_context_elements.llm_predictor import (
        BaseLLMPredictor,
    )
    from llama_index.core.service_context_elements.llm_predictor import (
        LLMPredictor,
    )
    from llama_index.core.tools.types import AsyncBaseTool
    from llama_index.core.tools.types import BaseTool
    from llama_index.core.tools.types import ToolMetadata
    from llama_index.core.vector_stores.types import VectorStore

    # These exist in the bridge but not here so define placeholders.
    RetrieverComponent = EmptyType

    from trulens.apps.llamaindex import WithFeedbackFilterNodes

else:
    # Otherwise llama_index is installed but is old so we have to use older imports.
    # Bridge for versions < 0.10

    if version is not None:
        logger.warning(
            "Using legacy llama_index version %s. Consider upgrading to 0.10.0 or later.",
            version,
        )

    from llama_index.chat_engine.types import AgentChatResponse
    from llama_index.chat_engine.types import BaseChatEngine
    from llama_index.chat_engine.types import StreamingAgentChatResponse
    from llama_index.core.base_query_engine import BaseQueryEngine
    from llama_index.core.base_query_engine import QueryEngineComponent
    from llama_index.core.base_retriever import BaseRetriever
    from llama_index.core.base_retriever import RetrieverComponent
    from llama_index.embeddings.base import BaseEmbedding
    from llama_index.indices.base import BaseIndex
    from llama_index.indices.prompt_helper import PromptHelper
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.llm_predictor import LLMPredictor
    from llama_index.llm_predictor.base import BaseLLMPredictor
    from llama_index.llm_predictor.base import LLMMetadata
    from llama_index.llms.base import BaseLLM
    from llama_index.memory import BaseMemory
    from llama_index.node_parser.interface import NodeParser
    from llama_index.postprocessor.types import BaseNodePostprocessor
    from llama_index.question_gen.types import BaseQuestionGenerator
    from llama_index.response.schema import Response
    from llama_index.response.schema import StreamingResponse
    from llama_index.response_synthesizers.base import BaseSynthesizer
    from llama_index.response_synthesizers.refine import Refine
    from llama_index.schema import BaseComponent
    from llama_index.tools.types import AsyncBaseTool
    from llama_index.tools.types import BaseTool
    from llama_index.tools.types import ToolMetadata
    from llama_index.vector_stores.types import VectorStore
    from trulens.apps.llamaindex import WithFeedbackFilterNodes


pp = PrettyPrinter()


def _retrieval_span() -> Dict[str, Union[SpanAttributes.SpanType, Attributes]]:
    def _attributes(ret, exception, *args, **kwargs) -> Attributes:
        attributes = {}
        # Guess query text.
        possible_query_texts = []
        for k, v in kwargs.items():
            if isinstance(v, str):
                possible_query_texts.append(v)
            elif isinstance(v, QueryBundle):
                possible_query_texts.append(v.query_str)
        # Guess retrieved contexts.
        retrieved_context = ret
        if isinstance(ret, list):
            if all(isinstance(curr, NodeWithScore) for curr in ret):
                retrieved_context = [curr.get_content() for curr in ret]
            elif all(hasattr(curr, "text") for curr in ret):
                retrieved_context = [curr.text for curr in ret]
        # Return.
        attributes = {
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: retrieved_context
        }
        if len(possible_query_texts) < 1:
            logger.info(
                "Could not guess query text for retrieval span as no likely candidates found!"
            )
        elif len(possible_query_texts) == 1:
            attributes[SpanAttributes.RETRIEVAL.QUERY_TEXT] = (
                possible_query_texts[0]
            )
        elif len(possible_query_texts) > 1:
            logger.info(
                "Could not guess query text for retrieval span! Found multiple possible query texts: %s",
                pp.pformat(possible_query_texts),
            )
        return attributes

    return {
        "span_type": SpanAttributes.SpanType.RETRIEVAL,
        "attributes": _attributes,
    }


class LlamaInstrument(core_instruments.Instrument):
    """Instrumentation for LlamaIndex apps."""

    class Default:
        """Instrumentation specification for LlamaIndex apps."""

        MODULES = {"llama_index.", "llama_hub."}.union(
            mod_tru_chain.LangChainInstrument.Default.MODULES
        )
        """Modules by prefix to instrument.

        Note that llama_index uses langchain internally for some things.
        """

        CLASSES = lambda: {
            BaseComponent,
            BaseLLM,
            BaseQueryEngine,
            BaseRetriever,
            BaseIndex,
            BaseChatEngine,
            BaseQuestionGenerator,
            BaseSynthesizer,
            Refine,
            LLMPredictor,
            LLMMetadata,
            BaseLLMPredictor,
            VectorStore,
            PromptHelper,
            BaseEmbedding,
            NodeParser,
            ToolMetadata,
            BaseTool,
            BaseMemory,
            WithFeedbackFilterNodes,
            BaseNodePostprocessor,
            QueryEngineComponent,
            RetrieverComponent,
        }.union(mod_tru_chain.LangChainInstrument.Default.CLASSES())
        """Classes to instrument."""

        METHODS: List[InstrumentedMethod] = (
            mod_tru_chain.LangChainInstrument.Default.METHODS
            + [
                InstrumentedMethod("chat", BaseLLM),
                InstrumentedMethod("complete", BaseLLM),
                InstrumentedMethod("stream_chat", BaseLLM),
                InstrumentedMethod("stream_complete", BaseLLM),
                InstrumentedMethod("achat", BaseLLM),
                InstrumentedMethod("acomplete", BaseLLM),
                InstrumentedMethod("astream_chat", BaseLLM),
                InstrumentedMethod("astream_complete", BaseLLM),
                InstrumentedMethod("__call__", BaseTool),
                InstrumentedMethod("call", BaseTool),
                InstrumentedMethod("acall", AsyncBaseTool),
                InstrumentedMethod("put", BaseMemory),
                InstrumentedMethod("get_response", Refine),
                InstrumentedMethod("predict", BaseLLMPredictor),
                InstrumentedMethod("apredict", BaseLLMPredictor),
                InstrumentedMethod("stream", BaseLLMPredictor),
                InstrumentedMethod("astream", BaseLLMPredictor),
                InstrumentedMethod("query", BaseQueryEngine),
                InstrumentedMethod("aquery", BaseQueryEngine),
                InstrumentedMethod("synthesize", BaseQueryEngine),
                InstrumentedMethod("asynthesize", BaseQueryEngine),
                InstrumentedMethod("chat", BaseChatEngine),
                InstrumentedMethod("achat", BaseChatEngine),
                InstrumentedMethod("stream_chat", BaseChatEngine),
                InstrumentedMethod("astream_chat", BaseChatEngine),
                InstrumentedMethod("complete", BaseChatEngine),
                InstrumentedMethod("acomplete", BaseChatEngine),
                InstrumentedMethod("stream_complete", BaseChatEngine),
                InstrumentedMethod("astream_complete", BaseChatEngine),
                InstrumentedMethod(
                    "retrieve",
                    BaseQueryEngine,
                    **_retrieval_span(),
                ),
                InstrumentedMethod(
                    "_retrieve",
                    BaseQueryEngine,
                    **_retrieval_span(),
                ),
                InstrumentedMethod(
                    "_aretrieve",
                    BaseQueryEngine,
                    **_retrieval_span(),
                ),
                InstrumentedMethod(
                    "retrieve",
                    BaseRetriever,
                    **_retrieval_span(),
                ),
                InstrumentedMethod(
                    "_retrieve",
                    BaseRetriever,
                    **_retrieval_span(),
                ),
                InstrumentedMethod(
                    "_aretrieve",
                    BaseRetriever,
                    **_retrieval_span(),
                ),
                InstrumentedMethod(
                    "retrieve",
                    WithFeedbackFilterNodes,
                    **_retrieval_span(),
                ),
                InstrumentedMethod(
                    "_retrieve",
                    WithFeedbackFilterNodes,
                    **_retrieval_span(),
                ),
                InstrumentedMethod(
                    "_aretrieve",
                    WithFeedbackFilterNodes,
                    **_retrieval_span(),
                ),
                InstrumentedMethod("_postprocess_nodes", BaseNodePostprocessor),
                InstrumentedMethod("_run_component", QueryEngineComponent),
                InstrumentedMethod("_run_component", RetrieverComponent),
            ]
        )
        """Methods to instrument."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LlamaInstrument.Default.MODULES,
            include_classes=LlamaInstrument.Default.CLASSES(),
            include_methods=LlamaInstrument.Default.METHODS,
            *args,
            **kwargs,
        )


class TruLlama(core_app.App):
    """Recorder for _LlamaIndex_ applications.

    This recorder is designed for LlamaIndex apps, providing a way to
    instrument, log, and evaluate their behavior.

    Example: "Creating a LlamaIndex application"

        Consider an example LlamaIndex application. For the complete code
        example, see [LlamaIndex
        Quickstart](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html).

        ```python
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)

        query_engine = index.as_query_engine()
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
        from trulens.apps.llamaindex import TruLlama
        context = TruLlama.select_context(query_engine)

        # Use feedback
        f_context_relevance = (
            Feedback(provider.context_relevance_with_context_reasons)
            .on_input()
            .on(context)  # Refers to context defined from `select_context`
            .aggregate(np.mean)
        )
        ```

    The application can be wrapped in a `TruLlama` recorder to provide logging
    and evaluation upon the application's use.

    Example: "Using the `TruLlama` recorder"

        ```python
        from trulens.apps.llamaindex import TruLlama
        # f_lang_match, f_qa_relevance, f_context_relevance are feedback functions
        tru_recorder = TruLlama(query_engine,
            app_name='LlamaIndex",
            app_version="base',
            feedbacks=[f_lang_match, f_qa_relevance, f_context_relevance])

        with tru_recorder as recording:
            query_engine.query("What is llama index?")
        ```

    Feedback functions can utilize the specific context produced by the
    application's query engine. This is achieved using the `select_context`
    method, which then can be used by a feedback selector, such as
    `on(context)`.

    Further information about LlamaIndex apps can be found on the [🦙 LlamaIndex
    Documentation](https://docs.llamaindex.ai/en/stable/) page.

    Args:
        app: A LlamaIndex application.

        **kwargs: Additional arguments to pass to [App][trulens.core.app.App]
            and [AppDefinition][trulens.core.schema.app.AppDefinition].
    """

    app: Union[BaseQueryEngine, BaseChatEngine]

    # TODEP
    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)

    def __init__(
        self,
        app: Union[BaseQueryEngine, BaseChatEngine],
        main_method: Optional[Callable] = None,
        **kwargs: dict,
    ):
        # TruLlama specific:
        kwargs["app"] = app
        # Create `TruSession` if not already created.
        if "connector" in kwargs:
            TruSession(connector=kwargs["connector"])
        else:
            TruSession()

        if main_method is not None:
            kwargs["main_method"] = main_method
        kwargs["root_class"] = pyschema_utils.Class.of_object(
            app
        )  # TODO: make class property
        kwargs["instrument"] = LlamaInstrument(app=self)

        super().__init__(**kwargs)

    @classmethod
    def select_source_nodes(cls) -> serial_utils.Lens:
        """
        Get the path to the source nodes in the query output.
        """
        return cls.select_outputs().source_nodes[:]

    # WithInstrumentCallbacks requirement:
    def wrap_lazy_values(
        self,
        rets: Any,
        wrap: Callable[[T], T],
        on_done: Optional[Callable[[T], T]],
        context_vars: Optional[python_utils.ContextVarsOrValues] = None,
    ) -> Any:
        """Wrap any llamaindex specific lazy values with wrappers that have callback wrap."""

        # NOTE(piotrm): This is all very frail. We need to make sure we call
        # on_done on things which are not lazy and wrap things which are lazy.
        # It is not easy to tell which is which sometimes in llamaindex.

        was_lazy = False

        members = {k: v for k, v in getmembers_static(rets)}

        if hasattr(rets, "is_done") and rets.is_done:
            return on_done(rets)

        if isinstance(rets, (Response)):
            return on_done(rets)

        if isinstance(rets, (AgentChatResponse)):
            return on_done(rets)

        if "async_response_gen" in members and isinstance(
            rets.async_response_gen, AsyncGenerator
        ):
            rets.async_response_gen = python_utils.wrap_async_generator(
                rets.async_response_gen,
                wrap=wrap,
                on_done=on_done,
                context_vars=context_vars,
            )
            was_lazy = True

        if "achat_stream" in members and isinstance(
            rets.achat_stream, AsyncGenerator
        ):
            rets.achat_stream = python_utils.wrap_async_generator(
                rets.achat_stream,
                wrap=wrap,
                on_done=on_done,
                context_vars=context_vars,
            )
            was_lazy = True

        if "chat_stream" in members and isinstance(rets.chat_stream, Generator):
            rets.chat_stream = python_utils.wrap_generator(
                rets.chat_stream,
                wrap=wrap,
                on_done=on_done,
                context_vars=context_vars,
            )
            was_lazy = True

        if "response_gen" in members and isinstance(
            rets.response_gen, Generator
        ):
            wrapped_response_gen = python_utils.wrap_generator(
                rets.response_gen,
                wrap=wrap,
                on_done=on_done,
                context_vars=context_vars,
            )
            if isinstance(members["response_gen"], property):
                # NOTE(piotrm): problem here as this is a property in
                # StramingAgentChatResponse so we cannot set it. Instead we override the
                # class which has an overridden property.

                # TODO(piotrm): Figure out if there is an easier way to override
                # an attribute which is a property.

                class Wrappable(rets.__class__):
                    @property
                    def response_gen(self):
                        return wrapped_response_gen

                Wrappable.__name__ = rets.__class__.__name__
                rets.__class__ = Wrappable

            else:
                rets.response_gen = wrapped_response_gen

            was_lazy = True

        if was_lazy:
            return rets

        else:
            return on_done(rets)

    # App override:
    @classmethod
    def select_context(
        cls, app: Optional[Union[BaseQueryEngine, BaseChatEngine]] = None
    ) -> serial_utils.Lens:
        """
        Get the path to the context in the query output.
        """
        return cls.select_outputs().source_nodes[:].node.text

    # App override:
    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """

        if "messages" in bindings.arguments:
            if len(bindings.arguments["messages"]) == 0:
                raise NotImplementedError(
                    "Cannot handle no messages in TruLens."
                )
            return str(bindings.arguments["messages"])

        if "str_or_query_bundle" in bindings.arguments:
            # llama_index specific
            str_or_bundle = bindings.arguments["str_or_query_bundle"]
            if isinstance(str_or_bundle, QueryBundle):
                return str_or_bundle.query_str
            else:
                return str_or_bundle

        elif "message" in bindings.arguments:
            # llama_index specific
            return bindings.arguments["message"]

        else:
            return core_app.App.main_input(self, func, sig, bindings)

    # App override:
    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> Optional[str]:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        try:
            attr = self._main_output_attribute(ret)

            if attr is not None:
                val = getattr(ret, attr)
                if attr == "response_txt":
                    if val is None:
                        # Return a placeholder value for now.
                        return (
                            f"TruLens: this app produced a streaming response of type {python_utils.class_name(type(ret))}. "
                            "The main output will not be available in TruLens."
                        )

                return val

            else:  # attr is None
                return core_app.App.main_output(self, func, sig, bindings, ret)

        except NotImplementedError:
            return None

    def _main_output_attribute(self, ret: Any) -> Optional[str]:
        """Which attribute in ret contains the main output of this llama_index
        app.
        """

        if isinstance(ret, Response):  # query, aquery
            return "response"

        elif isinstance(
            ret, (AgentChatResponse, StreamingAgentChatResponse)
        ):  #  chat, achat, stream_chat
            return "response"

        elif isinstance(
            ret,
            (StreamingResponse, AsyncStreamingResponse),
        ):
            return "response_txt"  # note that this is only available after the stream has been iterated over

        return None

    def main_call(self, human: str):
        # If available, a single text to a single text invocation of this app.
        # Note that we are ultimately calling a different langchain method here
        # than in the `main_acall` method so we don't reuse the async version
        # here in case langchain does something special between them.

        if isinstance(self.app, BaseQueryEngine):
            ret = self.app.query(human)
        elif isinstance(self.app, BaseChatEngine):
            ret = self.app.chat(human)
        else:
            raise RuntimeError(
                f"Do not know what the main method for app of type {type(self.app).__name__} is."
            )

        try:
            attr = self._main_output_attribute(ret)
            assert attr is not None
            return getattr(ret, attr)

        except Exception:
            raise NotImplementedError(
                f"Do not know what in object of type {type(ret).__name__} is the main app output."
            )

    async def main_acall(self, human: str):
        # If available, a single text to a single text invocation of this app.

        if isinstance(self.app, BaseQueryEngine):
            ret = await self.app.aquery(human)
        elif isinstance(self.app, BaseChatEngine):
            ret = await self.app.achat(human)
        else:
            raise RuntimeError(
                f"Do not know what the main async method for app of type {type(self.app).__name__} is."
            )

        try:
            attr = self._main_output_attribute(ret)
            assert attr is not None
            return getattr(ret, attr)

        except Exception:
            raise NotImplementedError(
                f"Do not know what in object of type {type(ret).__name__} is the main app output."
            )


TruLlama.model_rebuild()
