"""
# Llama_index instrumentation and monitoring. 
"""

from datetime import datetime
import logging
from pprint import PrettyPrinter
import traceback
from typing import ClassVar, Sequence, Tuple, Union, Callable, Any
from inspect import Signature, BoundArguments

from pydantic import Field

from trulens_eval.utils.llama import WithFeedbackFilterNodes
from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.provider_apis import Endpoint
from trulens_eval.provider_apis import OpenAIEndpoint
from trulens_eval.schema import Cost
from trulens_eval.schema import Record
from trulens_eval.schema import RecordAppCall
from trulens_eval.util import Class
from trulens_eval.util import dict_set_with
from trulens_eval.util import FunctionOrMethod
from trulens_eval.util import JSONPath
from trulens_eval.util import Method
from trulens_eval.util import OptionalImports
from trulens_eval.util import REQUIREMENT_LLAMA

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

with OptionalImports(message=REQUIREMENT_LLAMA):
    import llama_index
    from llama_index.indices.query.base import BaseQueryEngine
    from llama_index.chat_engine.types import BaseChatEngine
    from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
    from llama_index.response.schema import Response, StreamingResponse, RESPONSE_TYPE
    from llama_index.indices.query.schema import QueryBundle, QueryType

from trulens_eval.tru_chain import LangChainInstrument


class LlamaInstrument(Instrument):

    class Default:
        MODULES = {"llama_index."}.union(
            LangChainInstrument.Default.MODULES
        )  # NOTE: llama_index uses langchain internally for some things

        # Putting these inside thunk as llama_index is optional.
        CLASSES = lambda: {
            llama_index.indices.query.base.BaseQueryEngine,
            llama_index.indices.base_retriever.BaseRetriever,
            llama_index.indices.base.BaseIndex,
            llama_index.chat_engine.types.BaseChatEngine,
            llama_index.prompts.base.Prompt,
            # llama_index.prompts.prompt_type.PromptType, # enum
            llama_index.question_gen.types.BaseQuestionGenerator,
            llama_index.response_synthesizers.base.BaseSynthesizer,
            llama_index.response_synthesizers.refine.Refine,
            llama_index.llm_predictor.LLMPredictor,
            llama_index.llm_predictor.base.LLMMetadata,
            llama_index.llm_predictor.base.BaseLLMPredictor,
            llama_index.vector_stores.types.VectorStore,
            llama_index.indices.service_context.ServiceContext,
            llama_index.indices.prompt_helper.PromptHelper,
            llama_index.embeddings.base.BaseEmbedding,
            llama_index.node_parser.interface.NodeParser,
            WithFeedbackFilterNodes
        }.union(LangChainInstrument.Default.CLASSES())

        # Instrument only methods with these names and of these classes. Ok to
        # include llama_index inside methods.
        METHODS = dict_set_with(
            {
                "get_response":
                    lambda o: isinstance(
                        o, llama_index.response_synthesizers.refine.Refine
                    ),
                "predict":
                    lambda o: isinstance(
                        o, llama_index.llm_predictor.base.BaseLLMPredictor
                    ),
                "query":
                    lambda o: isinstance(
                        o, llama_index.indices.query.base.BaseQueryEngine
                    ),
                "aquery":
                    lambda o: isinstance(
                        o, llama_index.indices.query.base.BaseQueryEngine
                    ),
                "chat":
                    lambda o:
                    isinstance(o, llama_index.chat_engine.types.BaseChatEngine),
                "achat":
                    lambda o:
                    isinstance(o, llama_index.chat_engine.types.BaseChatEngine),
                "stream_chat":
                    lambda o:
                    isinstance(o, llama_index.chat_engine.types.BaseChatEngine),
                "astream_achat":
                    lambda o:
                    isinstance(o, llama_index.chat_engine.types.BaseChatEngine),
                "retrieve":
                    lambda o: isinstance(
                        o, (
                            llama_index.indices.query.base.BaseQueryEngine,
                            llama_index.indices.base_retriever.BaseRetriever,
                            WithFeedbackFilterNodes
                        )
                    ),
                "synthesize":
                    lambda o: isinstance(
                        o, llama_index.indices.query.base.BaseQueryEngine
                    ),
            }, LangChainInstrument.Default.METHODS
        )

    def __init__(self, *args, **kwargs):
        super().__init__(
            include_modules=LlamaInstrument.Default.MODULES,
            include_classes=LlamaInstrument.Default.CLASSES(),
            include_methods=LlamaInstrument.Default.METHODS,
            *args, **kwargs
        )


class TruLlama(App):
    """
    Wrap a llama index engine for monitoring.

    Arguments:
    - app: BaseQueryEngine | BaseChatEngine -- the engine to wrap.
    - More args in App
    - More args in AppDefinition
    - More args in WithClassInfo
    """

    class Config:
        arbitrary_types_allowed = True

    app: Union[BaseQueryEngine, BaseChatEngine]

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruLlama.query),
        const=True
    )

    def __init__(self, app: BaseQueryEngine, **kwargs):

        super().update_forward_refs()

        # TruLlama specific:
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)  # TODO: make class property
        kwargs['instrument'] = LlamaInstrument(
            root_methods=set(
                [
                    TruLlama.with_record, TruLlama.awith_record,
                ]
            ),
            callbacks=self
        )

        super().__init__(**kwargs)

        self.post_init()

    @classmethod
    def select_source_nodes(cls) -> JSONPath:
        """
        Get the path to the source nodes in the query output.
        """
        return cls.select_outputs().source_nodes[:]

    # llama_index.chat_engine.types.BaseChatEngine
    def chat(self, *args, **kwargs) -> AgentChatResponse:
        assert isinstance(
            self.app, llama_index.chat_engine.types.BaseChatEngine
        )

        res, _ = self.chat_with_record(*args, **kwargs)
        return res

    # llama_index.chat_engine.types.BaseChatEngine
    async def achat(self, *args, **kwargs) -> AgentChatResponse:
        assert isinstance(
            self.app, llama_index.chat_engine.types.BaseChatEngine
        )

        res, _ = await self.achat_with_record(*args, **kwargs)
        return res

    # llama_index.chat_engine.types.BaseChatEngine
    def stream_chat(self, *args, **kwargs) -> StreamingAgentChatResponse:
        assert isinstance(
            self.app, llama_index.chat_engine.types.BaseChatEngine
        )

        res, _ = self.stream_chat_with_record(*args, **kwargs)
        return res

    # llama_index.chat_engine.types.BaseChatEngine
    async def astream_chat(self, *args, **kwargs) -> StreamingAgentChatResponse:
        assert isinstance(
            self.app, llama_index.chat_engine.types.BaseChatEngine
        )

        res, _ = await self.astream_chat_with_record(*args, **kwargs)
        return res

    # llama_index.indices.query.base.BaseQueryEngine
    def query(self, *args, **kwargs) -> RESPONSE_TYPE:
        assert isinstance(
            self.app, llama_index.indices.query.base.BaseQueryEngine
        )

        res, _ = self.query_with_record(*args, **kwargs)
        return res

    # llama_index.indices.query.base.BaseQueryEngine
    async def aquery(self, *args, **kwargs) -> RESPONSE_TYPE:
        assert isinstance(
            self.app, llama_index.indices.query.base.BaseQueryEngine
        )

        res, _ = await self.aquery_with_record(*args, **kwargs)
        return res

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
            return bindings.arguments['str_or_query_bundle']
        
        elif 'message' in bindings.arguments:
            # llama_index specific
            return bindings.arguments['message']

        else:

            return App.main_input(self, func, sig, bindings)

    def main_output(
        self, func: Callable, sig: Signature, bindings: BoundArguments, ret: Any
    ) -> str:
        """
        Determine the main out string for the given function `func` with
        signature `sig` after it is called with the given `bindings` and has
        returned `ret`.
        """

        if isinstance(ret, Response): # query, aquery
            return ret.response

        elif isinstance(ret, AgentChatResponse): #  chat, achat
            return ret.response

        elif isinstance(ret, (StreamingResponse, StreamingAgentChatResponse)):
            logger.warn(
                "App produced a streaming response. "
                "Tracking content of streams in llama_index is not yet supported. "
                "App main_output will be None."
            )

            return None

        else:

            return App.main_output(self, func, sig, bindings, ret)

    # Mirrors llama_index.indices.query.base.BaseQueryEngine.query .
    def query_with_record(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[RESPONSE_TYPE, Record]:
        assert isinstance(
            self.app, llama_index.indices.query.base.BaseQueryEngine
        )

        return self.with_record(self.app.query, str_or_query_bundle)


    # Mirrors llama_index.indices.query.base.BaseQueryEngine.aquery .
    async def aquery_with_record(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[RESPONSE_TYPE, Record]:
        assert isinstance(
            self.app, llama_index.indices.query.base.BaseQueryEngine
        )

        return await self.awith_record(self.app.aquery, str_or_query_bundle)


    # Compatible with llama_index.chat_engine.types.BaseChatEngine.chat .
    def chat_with_record(self, message: str,
                         **kwargs) -> Tuple[AgentChatResponse, Record]:
        assert isinstance(
            self.app, llama_index.chat_engine.types.BaseChatEngine
        )

        return self.with_record(self.app.chat, message, **kwargs)


    # Compatible with llama_index.chat_engine.types.BaseChatEngine.achat .
    async def achat_with_record(self, message: str,
                                **kwargs) -> Tuple[AgentChatResponse, Record]:
        assert isinstance(
            self.app, llama_index.chat_engine.types.BaseChatEngine
        )

        return await self.awith_record(self.app.achat, message, **kwargs)


    # Compatible with llama_index.chat_engine.types.BaseChatEngine.stream_chat .
    def stream_chat_with_record(
        self, message: str, **kwargs
    ) -> Tuple[StreamingAgentChatResponse, Record]:
        assert isinstance(
            self.app, llama_index.chat_engine.types.BaseChatEngine
        )

        return self.with_record(self.app.stream_chat, message, **kwargs)


    # Compatible with llama_index.chat_engine.types.BaseChatEngine.astream_chat .
    async def astream_chat_with_record(
        self, message: str, **kwargs
    ) -> Tuple[StreamingAgentChatResponse, Record]:
        assert isinstance(
            self.app, llama_index.chat_engine.types.BaseChatEngine
        )

        return await self.awith_record(self.app.astream_chat, message, **kwargs)
