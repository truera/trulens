"""
# Llama_index instrumentation and monitoring. 
"""

from datetime import datetime
import logging
from pprint import PrettyPrinter
from typing import Sequence, Tuple

from trulens_eval.instruments import Instrument
from trulens_eval.schema import Record
from trulens_eval.schema import RecordAppCall
from trulens_eval.tru_app import TruApp
from trulens_eval.util import Class
from trulens_eval.util import OptionalImports
from trulens_eval.util import REQUIREMENT_LLAMA
from trulens_eval.utils.llama import Is

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

with OptionalImports(message=REQUIREMENT_LLAMA):
    import llama_index
    from llama_index.indices.query.base import BaseQueryEngine
    from llama_index.response.schema import Response


class LlamaInstrument(Instrument):

    class Default:
        MODULES = {"llama_index."}

        # Putting these inside thunk as llama_index is optional.
        CLASSES = lambda: {
            llama_index.indices.query.base.BaseQueryEngine, llama_index.indices.
            base_retriever.BaseRetriever
            # query_engine.retriever_query_engine.RetrieverQueryEngine
        }

        # Instrument only methods with these names and of these classes. Ok to
        # include llama_index inside methods.
        METHODS = {
            "query":
                lambda o:
                isinstance(o, llama_index.indices.query.base.BaseQueryEngine),
            "retrieve":
                lambda o: isinstance(
                    o, (
                        llama_index.indices.query.base.BaseQueryEngine,
                        llama_index.indices.base_retriever.BaseRetriever
                    )
                ),
            "synthesize":
                lambda o:
                isinstance(o, llama_index.indices.query.base.BaseQueryEngine),
        }

    def __init__(self):
        super().__init__(
            root_method=TruLlama.query_with_record,
            modules=LlamaInstrument.Default.MODULES,
            classes=LlamaInstrument.Default.CLASSES(),  # was thunk
            methods=LlamaInstrument.Default.METHODS
        )


class TruLlama(TruApp):
    """
    Wrap a llama index engine for monitoring.

    Arguments:
    - app: RetrieverQueryEngine -- the engine to wrap.
    - More args in TruApp
    - More args in WithClassInfo
    """

    class Config:
        arbitrary_types_allowed = True

    app: BaseQueryEngine

    def __init__(self, app: BaseQueryEngine, **kwargs):

        super().update_forward_refs()

        # TruLlama specific:
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = LlamaInstrument()

        super().__init__(**kwargs)

    def query(self, *args, **kwargs) -> Response:
        res, _ = self.query_with_record(*args, **kwargs)

        return res

    def query_with_record(self, str_or_query_bundle) -> Tuple[Response, Record]:
        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record: Sequence[RecordAppCall] = []

        ret = None
        error = None

        total_tokens = None
        total_cost = None

        start_time = None
        end_time = None

        try:
            # TODO: do this only if there is an openai model inside the app:
            # with get_openai_callback() as cb:
            start_time = datetime.now()
            ret = self.app.query(str_or_query_bundle)
            end_time = datetime.now()
            # total_tokens = cb.total_tokens
            # total_cost = cb.total_cost
            total_tokens = 0
            total_cost = 0

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
            ret_record_args, error, total_tokens, total_cost, start_time,
            end_time, record
        )

        return ret, ret_record

    def instrumented(self):
        return super().instrumented(categorizer=Is.what)
