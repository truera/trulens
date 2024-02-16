"""
# Llama_index instrumentation and monitoring. 
"""

from inspect import BoundArguments
from inspect import Signature
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Optional, Union

from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.utils.containers import dict_set_with
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LLAMA
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import Lens

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

with OptionalImports(messages=REQUIREMENT_LLAMA):
    import llama_index
    from llama_index.chat_engine.types import AgentChatResponse
    from llama_index.chat_engine.types import BaseChatEngine
    from llama_index.chat_engine.types import StreamingAgentChatResponse
    from llama_index.embeddings.base import BaseEmbedding
    from llama_index.indices.base import BaseIndex
    # retriever
    from llama_index.core.base_retriever import BaseRetriever
    from llama_index.core.base_retriever import RetrieverComponent
    # misc
    from llama_index.indices.prompt_helper import PromptHelper
    from llama_index.core.base_query_engine import BaseQueryEngine
    from llama_index.core.base_query_engine import QueryEngineComponent
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.indices.service_context import ServiceContext
    from llama_index.llm_predictor import LLMPredictor
    from llama_index.llm_predictor.base import BaseLLMPredictor
    from llama_index.llm_predictor.base import LLMMetadata
    from llama_index.postprocessor.types import BaseNodePostprocessor
    # LLMs
    from llama_index.llms.base import BaseLLM  # subtype of BaseComponent
    # memory
    from llama_index.memory import BaseMemory
    from llama_index.node_parser.interface import NodeParser
    from llama_index.prompts.base import Prompt
    from llama_index.question_gen.types import BaseQuestionGenerator
    from llama_index.response.schema import Response
    from llama_index.response.schema import StreamingResponse
    from llama_index.response_synthesizers.base import BaseSynthesizer
    from llama_index.response_synthesizers.refine import Refine
    from llama_index.schema import BaseComponent
    # agents
    from llama_index.tools.types import AsyncBaseTool  # subtype of BaseTool
    from llama_index.tools.types import BaseTool
    from llama_index.tools.types import \
        ToolMetadata  # all of the readable info regarding tools is in this class
    from llama_index.vector_stores.types import VectorStore

    from trulens_eval.utils.llama import WithFeedbackFilterNodes

# Need to `from ... import ...` for the below as referring to some of these
# later in this file by full path does not work due to lack of intermediate
# modules in the path.

# Fail outside of optional imports contexts so that anything that follows gets
# to be a dummy which will cause failures if used.
OptionalImports(messages=REQUIREMENT_LLAMA).assert_installed(llama_index)

from trulens_eval.tru_chain import LangChainInstrument


class LlamaInstrument(Instrument):

    class Default:
        MODULES = {"llama_index.", "llama_hub."}.union(
            LangChainInstrument.Default.MODULES
        )  # NOTE: llama_index uses langchain internally for some things

        # Putting these inside thunk as llama_index is optional.
        CLASSES = lambda: {
            BaseComponent,
            BaseLLM,
            BaseQueryEngine,
            BaseRetriever,
            BaseIndex,
            BaseChatEngine,
            Prompt,
            # llama_index.prompts.prompt_type.PromptType, # enum
            BaseQuestionGenerator,
            BaseSynthesizer,
            Refine,
            LLMPredictor,
            LLMMetadata,
            BaseLLMPredictor,
            VectorStore,
            ServiceContext,
            PromptHelper,
            BaseEmbedding,
            NodeParser,
            ToolMetadata,
            BaseTool,
            BaseMemory,
            WithFeedbackFilterNodes,
            BaseNodePostprocessor,
            QueryEngineComponent
        }.union(LangChainInstrument.Default.CLASSES())

        # Instrument only methods with these names and of these classes. Ok to
        # include llama_index inside methods.
        METHODS = dict_set_with(
            {
                # LLM:
                "complete":
                    lambda o: isinstance(o, BaseLLM),
                "stream_complete":
                    lambda o: isinstance(o, BaseLLM),
                "acomplete":
                    lambda o: isinstance(o, BaseLLM),
                "astream_complete":
                    lambda o: isinstance(o, BaseLLM),

                # BaseTool/AsyncBaseTool:
                "__call__":
                    lambda o: isinstance(o, BaseTool),
                "call":
                    lambda o: isinstance(o, BaseTool),
                "acall":
                    lambda o: isinstance(o, AsyncBaseTool),

                # Memory:
                "put":
                    lambda o: isinstance(o, BaseMemory),

                # Misc.:
                "get_response":
                    lambda o: isinstance(o, Refine),
                "predict":
                    lambda o: isinstance(o, BaseLLMPredictor),

                # BaseQueryEngine:
                "query":
                    lambda o: isinstance(o, BaseQueryEngine),
                "aquery":
                    lambda o: isinstance(o, BaseQueryEngine),

                # BaseChatEngine/LLM:
                "chat":
                    lambda o: isinstance(o, (BaseLLM, BaseChatEngine)),
                "achat":
                    lambda o: isinstance(o, (BaseLLM, BaseChatEngine)),
                "stream_chat":
                    lambda o: isinstance(o, (BaseLLM, BaseChatEngine)),
                "astream_achat":
                    lambda o: isinstance(o, (BaseLLM, BaseChatEngine)),

                # BaseRetriever/BaseQueryEngine:
                "retrieve":
                    lambda o: isinstance(
                        o, (
                            BaseQueryEngine, BaseRetriever,
                            WithFeedbackFilterNodes
                        )
                    ),
                "_retrieve":
                    lambda o: isinstance(
                        o, (
                            BaseQueryEngine, BaseRetriever,
                            WithFeedbackFilterNodes
                        )
                    ),
                "_aretrieve":
                    lambda o: isinstance(
                        o, (
                            BaseQueryEngine, BaseRetriever,
                            WithFeedbackFilterNodes
                        )
                    ),
                # BaseQueryEngine:
                "synthesize":
                    lambda o: isinstance(o, BaseQueryEngine),

                # BaseNodePostProcessor
                "_postprocess_nodes":
                    lambda o: isinstance(o, (BaseNodePostprocessor)),

                # Components
                "_run_component":
                    lambda o: isinstance(o, (QueryEngineComponent, RetrieverComponent))
            },
            LangChainInstrument.Default.METHODS
        )

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LlamaInstrument.Default.MODULES,
            include_classes=LlamaInstrument.Default.CLASSES(),
            include_methods=LlamaInstrument.Default.METHODS,
            *args,
            **kwargs
        )


class TruLlama(App):
    """
    Instantiates the LLama Index Wrapper.

    Example:
        LLama-Index code: [LLama Index
        Quickstart](https://gpt-index.readthedocs.io/en/stable/getting_started/starter_example.html)
    
        ```python
        # Code snippet taken from llama_index 0.8.29 (API subject to change with new versions)
        from llama_index import VectorStoreIndex
        from llama_index.readers.web import SimpleWebPageReader

        documents = SimpleWebPageReader(
            html_to_text=True
        ).load_data(["http://paulgraham.com/worked.html"])
        index = VectorStoreIndex.from_documents(documents)

        query_engine = index.as_query_engine()
        ```

        Trulens Eval Code:
        ```python
        from trulens_eval import TruLlama
        # f_lang_match, f_qa_relevance, f_qs_relevance are feedback functions
        tru_recorder = TruLlama(query_engine,
            app_id='LlamaIndex_App1',
            feedbacks=[f_lang_match, f_qa_relevance, f_qs_relevance])

        with tru_recorder as recording:
            query_engine.query("What is llama index?")

        tru_record = recording.records[0]

        # To add record metadata 
        with tru_recorder as recording:
            recording.record_metadata="this is metadata for all records in this context that follow this line"
            query_engine.query("What is llama index?")
            recording.record_metadata="this is different metadata for all records in this context that follow this line"
            query_engine.query("Where do I download llama index?")
        
        ```

        See [Feedback Functions](https://www.trulens.org/trulens_eval/api/feedback/) for instantiating feedback functions.

    Args:
        app: A llama index application.

        **kwargs: Additional arguments to pass to [App][trulens_eval.app.App]
            and [AppDefinition][trulens_eval.app.AppDefinition]
    """

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    app: Union[BaseQueryEngine, BaseChatEngine]

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruLlama.query)
    )

    def __init__(self, app: Union[BaseQueryEngine, BaseChatEngine], **kwargs: dict):
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

            return App.main_input(self, func, sig, bindings)

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
                return App.main_output(self, func, sig, bindings, ret)

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


TruLlama.model_rebuild()
