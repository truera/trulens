"""
# Llama_index instrumentation and monitoring. 
"""

from datetime import datetime
import logging
from pprint import PrettyPrinter
from typing import ClassVar, Sequence, Tuple

from pydantic import Field

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
    from llama_index.response.schema import Response

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
            llama_index.node_parser.interface.NodeParser
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
                "retrieve":
                    lambda o: isinstance(
                        o, (
                            llama_index.indices.query.base.BaseQueryEngine,
                            llama_index.indices.base_retriever.BaseRetriever
                        )
                    ),
                "synthesize":
                    lambda o: isinstance(
                        o, llama_index.indices.query.base.BaseQueryEngine
                    ),
            }, LangChainInstrument.Default.METHODS
        )

    def __init__(self):
        super().__init__(
            root_method=TruLlama.query_with_record,
            modules=LlamaInstrument.Default.MODULES,
            classes=LlamaInstrument.Default.CLASSES(),  # was thunk
            methods=LlamaInstrument.Default.METHODS
        )


class TruLlama(App):
    """
    Wrap a llama index engine for monitoring.

    Arguments:
    - app: RetrieverQueryEngine -- the engine to wrap.
    - More args in App
    - More args in AppDefinition
    - More args in WithClassInfo
    """

    class Config:
        arbitrary_types_allowed = True

    app: BaseQueryEngine

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruLlama.query),
        const=True
    )

    def __init__(self, app: BaseQueryEngine, **kwargs):

        super().update_forward_refs()

        # TruLlama specific:
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)  # TODO: make class property
        kwargs['instrument'] = LlamaInstrument()

        super().__init__(**kwargs)

    def query(self, *args, **kwargs) -> Response:
        res, _ = self.query_with_record(*args, **kwargs)
        return res

    @classmethod
    def select_source_nodes(cls) -> JSONPath:
        """
        Get the path to the source nodes in the query output.
        """
        return cls.select_outputs().source_nodes[:]

    def query_with_record(self, str_or_query_bundle) -> Tuple[Response, Record]:
        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record: Sequence[RecordAppCall] = []

        ret = None
        error = None

        start_time = None
        end_time = None

        cost = Cost()

        try:
            start_time = datetime.now()
            ret, cost = Endpoint.track_all_costs_tally(
                lambda: self.app.query(str_or_query_bundle)
            )

            end_time = datetime.now()

        except BaseException as e:
            end_time = datetime.now()
            error = e
            logger.error(f"Engine raised an exception: {e}")

        assert len(record) > 0, "No information recorded in call."

        ret_record_args = dict()

        # TODO: generalize
        ret_record_args['main_input'] = str_or_query_bundle
        if ret is not None:
            # TODO: generalize and error check
            ret_record_args['main_output'] = ret.response

        ret_record = self._post_record(
            ret_record_args, error, cost, start_time, end_time, record
        )

        return ret, ret_record
