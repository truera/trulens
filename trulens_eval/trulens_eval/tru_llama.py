"""
# LlamaIndex instrumentation.
"""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Dict, Optional, Union

from pydantic import Field

from trulens_eval import app as mod_app
from trulens_eval.instruments import ClassFilter
from trulens_eval.instruments import Instrument
from trulens_eval.utils.containers import dict_set_with_multikey
from trulens_eval.utils.imports import Dummy
from trulens_eval.utils.imports import get_package_version
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import parse_version
from trulens_eval.utils.imports import REQUIREMENT_LLAMA
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.python import EmptyType
from trulens_eval.utils.serial import Lens

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

with OptionalImports(messages=REQUIREMENT_LLAMA) as opt:
    import llama_index

    version = get_package_version("llama_index")

    # If llama index is not installed, will get a dummy for llama_index. In that
    # case or if it is installed and sufficiently new version, continue with
    # this set of imports. We don't want to partially run the new imports and
    # fail midway to continue with the legacy imports.

    legacy: bool = version is None or isinstance(llama_index, Dummy)

    if not legacy:
        # Check if llama_index is new enough.

        if version < parse_version("0.10.0"):
            legacy = True

    if not legacy:
        from llama_index.core.base.base_query_engine import BaseQueryEngine
        from llama_index.core.base.base_query_engine import \
            QueryEngineComponent
        from llama_index.core.base.embeddings.base import BaseEmbedding
        from llama_index.core.base.llms.base import BaseLLM
        from llama_index.core.base.llms.types import LLMMetadata
        from llama_index.core.base.response.schema import Response
        from llama_index.core.base.response.schema import StreamingResponse
        from llama_index.core.chat_engine.types import AgentChatResponse
        from llama_index.core.chat_engine.types import BaseChatEngine
        from llama_index.core.chat_engine.types import \
            StreamingAgentChatResponse
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
        from llama_index.core.schema import QueryBundle
        from llama_index.core.tools.types import AsyncBaseTool
        from llama_index.core.tools.types import BaseTool
        from llama_index.core.tools.types import ToolMetadata
        from llama_index.core.vector_stores.types import VectorStore
        from llama_index.legacy.llm_predictor import LLMPredictor
        from llama_index.legacy.llm_predictor.base import BaseLLMPredictor

        # These exist in the bridge but not here so define placeholders.
        RetrieverComponent = EmptyType

        from trulens_eval.guardrails.llama import WithFeedbackFilterNodes

    else:
        # Otherwise llama_index is installed but is old so we have to use older imports.
        # Bridge for versions < 0.10

        if version is not None:
            logger.warning(
                "Using legacy llama_index version %s. Consider upgrading to 0.10.0 or later.",
                version
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
        from llama_index.indices.service_context import ServiceContext
        from llama_index.llm_predictor import LLMPredictor
        from llama_index.llm_predictor.base import BaseLLMPredictor
        from llama_index.llm_predictor.base import LLMMetadata
        from llama_index.llms.base import BaseLLM
        from llama_index.memory import BaseMemory
        from llama_index.node_parser.interface import NodeParser
        from llama_index.postprocessor.types import BaseNodePostprocessor
        from llama_index.prompts.base import Prompt
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

        from trulens_eval.utils.llama import WithFeedbackFilterNodes

# Need to `from ... import ...` for the below as referring to some of these
# later in this file by full path does not work due to lack of intermediate
# modules in the path.

# Fail outside of optional imports contexts so that anything that follows gets
# to be a dummy which will cause failures if used.
opt.assert_installed(llama_index)

from trulens_eval.tru_chain import LangChainInstrument


class LlamaInstrument(Instrument):
    """Instrumentation for LlamaIndex apps."""

    class Default:
        """Instrumentation specification for LlamaIndex apps."""

        MODULES = {"llama_index.",
                   "llama_hub."}.union(LangChainInstrument.Default.MODULES)
        """Modules by prefix to instrument.
         
        Note that llama_index uses langchain internally for some things.
        """

        CLASSES = lambda: {
            BaseComponent, BaseLLM, BaseQueryEngine, BaseRetriever, BaseIndex,
            BaseChatEngine, BaseQuestionGenerator, BaseSynthesizer, Refine,
            LLMPredictor, LLMMetadata, BaseLLMPredictor, VectorStore,
            PromptHelper, BaseEmbedding, NodeParser, ToolMetadata, BaseTool,
            BaseMemory, WithFeedbackFilterNodes, BaseNodePostprocessor,
            QueryEngineComponent, RetrieverComponent
        }.union(LangChainInstrument.Default.CLASSES())
        """Classes to instrument."""

        METHODS: Dict[str, ClassFilter] = dict_set_with_multikey(
            dict(LangChainInstrument.Default.METHODS),
            {
                # LLM:
                (
                    "complete", "stream_complete", "acomplete", "astream_complete"
                ):
                    BaseLLM,

                # BaseTool/AsyncBaseTool:
                ("__call__", "call"):
                    BaseTool,
                ("acall"):
                    AsyncBaseTool,

                # Memory:
                ("put"):
                    BaseMemory,

                # Misc.:
                ("get_response"):
                    Refine,
                ("predict"):
                    BaseLLMPredictor,

                # BaseQueryEngine:
                ("query", "aquery"):
                    BaseQueryEngine,

                # BaseChatEngine/LLM:
                ("chat", "achat", "stream_chat", "astream_achat"):
                    (BaseLLM, BaseChatEngine),

                # BaseRetriever/BaseQueryEngine:
                ("retrieve", "_retrieve", "_aretrieve"):
                    (BaseQueryEngine, BaseRetriever, WithFeedbackFilterNodes),

                # BaseQueryEngine:
                ("synthesize"):
                    BaseQueryEngine,

                # BaseNodePostProcessor
                ("_postprocess_nodes"):
                    BaseNodePostprocessor,

                # Components
                ("_run_component"): (QueryEngineComponent, RetrieverComponent)
            }
        )
        """Methods to instrument."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LlamaInstrument.Default.MODULES,
            include_classes=LlamaInstrument.Default.CLASSES(),
            include_methods=LlamaInstrument.Default.METHODS,
            *args,
            **kwargs
        )


class TruLlama(mod_app.App):
    """Recorder for _LlamaIndex_ applications.

    This recorder is designed for LlamaIndex apps, providing a way to
    instrument, log, and evaluate their behavior.

    !!! example "Creating a LlamaIndex application"

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

    !!! example "Defining a feedback function"

        ```python
        from trulens_eval.feedback.provider import OpenAI
        from trulens_eval import Feedback
        import numpy as np

        # Select context to be used in feedback.
        from trulens_eval.app import App
        context = App.select_context(rag_chain)

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

    !!! example "Using the `TruLlama` recorder"

        ```python
        from trulens_eval import TruLlama
        # f_lang_match, f_qa_relevance, f_qs_relevance are feedback functions
        tru_recorder = TruLlama(query_engine,
            app_id='LlamaIndex_App1',
            feedbacks=[f_lang_match, f_qa_relevance, f_qs_relevance])

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

        **kwargs: Additional arguments to pass to [App][trulens_eval.app.App]
            and [AppDefinition][trulens_eval.schema.app.AppDefinition].
    """
    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    app: Union[BaseQueryEngine, BaseChatEngine]

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruLlama.query)
    )

    def __init__(
        self, app: Union[BaseQueryEngine, BaseChatEngine], **kwargs: dict
    ):
        # TruLlama specific:
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)  # TODO: make class property
        kwargs['instrument'] = LlamaInstrument(app=self)

        super().__init__(**kwargs)

    @classmethod
    def select_source_nodes(cls) -> Lens:
        """
        Get the path to the source nodes in the query output.
        """
        return cls.select_outputs().source_nodes[:]

    @classmethod
    def select_context(
        cls,
        app: Optional[Union[BaseQueryEngine, BaseChatEngine]] = None
    ) -> Lens:
        """
        Get the path to the context in the query output.
        """
        return cls.select_outputs().source_nodes[:].node.text

    def main_input(
        self, func: Callable, sig: Signature, bindings: BoundArguments
    ) -> str:
        """
        Determine the main input string for the given function `func` with
        signature `sig` if it is to be called with the given bindings
        `bindings`.
        """

        if 'str_or_query_bundle' in bindings.arguments:
            # llama_index specific
            str_or_bundle = bindings.arguments['str_or_query_bundle']
            if isinstance(str_or_bundle, QueryBundle):
                return str_or_bundle.query_str
            else:
                return str_or_bundle

        elif 'message' in bindings.arguments:
            # llama_index specific
            return bindings.arguments['message']

        else:

            return mod_app.App.main_input(self, func, sig, bindings)

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
                return getattr(ret, attr)
            else:  # attr is None
                return mod_app.App.main_output(self, func, sig, bindings, ret)

        except NotImplementedError:
            return None

    def _main_output_attribute(self, ret: Any) -> Optional[str]:
        """
        Which attribute in ret contains the main output of this llama_index app.
        """

        if isinstance(ret, Response):  # query, aquery
            return "response"

        elif isinstance(ret, AgentChatResponse):  #  chat, achat
            return "response"

        elif isinstance(ret, (StreamingResponse, StreamingAgentChatResponse)):
            raise NotImplementedError(
                "App produced a streaming response. "
                "Tracking content of streams in llama_index is not yet supported. "
                "App main_output will be None."
            )

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

    # TOREMOVE
    # llama_index.chat_engine.types.BaseChatEngine
    def chat(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="chat", is_async=False, with_record=False
        )

    # TOREMOVE
    # llama_index.chat_engine.types.BaseChatEngine
    async def achat(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="achat", is_async=True, with_record=False
        )

    # TOREMOVE
    # llama_index.chat_engine.types.BaseChatEngine
    def stream_chat(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="stream_chat", is_async=False, with_record=False
        )

    # TOREMOVE
    # llama_index.chat_engine.types.BaseChatEngine
    async def astream_chat(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="astream_chat", is_async=True, with_record=False
        )

    # TOREMOVE
    # llama_index.indices.query.base.BaseQueryEngine
    def query(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="query", is_async=False, with_record=False
        )

    # TOREMOVE
    # llama_index.indices.query.base.BaseQueryEngine
    async def aquery(self, *args, **kwargs) -> None:
        self._throw_dep_message(
            method="aquery", is_async=True, with_record=False
        )

    # TOREMOVE
    # Mirrors llama_index.indices.query.base.BaseQueryEngine.query .
    def query_with_record(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="query", is_async=False, with_record=True
        )

    # TOREMOVE
    # Mirrors llama_index.indices.query.base.BaseQueryEngine.aquery .
    async def aquery_with_record(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="aquery", is_async=True, with_record=True
        )

    # TOREMOVE
    # Compatible with llama_index.chat_engine.types.BaseChatEngine.chat .
    def chat_with_record(self, *args, **kwargs) -> None:

        self._throw_dep_message(method="chat", is_async=False, with_record=True)

    # TOREMOVE
    # Compatible with llama_index.chat_engine.types.BaseChatEngine.achat .
    async def achat_with_record(self, *args, **kwargs) -> None:

        self._throw_dep_message(method="achat", is_async=True, with_record=True)

    # TOREMOVE
    # Compatible with llama_index.chat_engine.types.BaseChatEngine.stream_chat .
    def stream_chat_with_record(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="stream", is_async=False, with_record=True
        )

    # TOREMOVE
    # Compatible with llama_index.chat_engine.types.BaseChatEngine.astream_chat .
    async def astream_chat_with_record(self, *args, **kwargs) -> None:

        self._throw_dep_message(
            method="astream_chat", is_async=True, with_record=True
        )


import trulens_eval  # for App class annotations

TruLlama.model_rebuild()
