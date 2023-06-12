"""
# Llama_index instrumentation and monitoring. 
"""

from collections import defaultdict
from datetime import datetime
from inspect import BoundArguments
from inspect import signature
from inspect import stack
import logging
import os
from pprint import PrettyPrinter
import threading as th
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


from pydantic import BaseModel
from pydantic import Field

from trulens_eval.instruments import Instrument
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import Method
from trulens_eval.schema import Query
from trulens_eval.schema import Record
from trulens_eval.schema import RecordChainCall
from trulens_eval.schema import RecordChainCallMethod
from trulens_eval.tru import Tru
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.tru_model import TruModel
from trulens_eval.util import Class
from trulens_eval.util import get_local_in_call_stack
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.util import noserio, REQUIREMENT_LLAMA
from trulens_eval.util import TP, import_optional
from trulens_eval.utils.llama import Is

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

llama_index = import_optional(mod="llama_index", message=REQUIREMENT_LLAMA)
query_engine = import_optional(mod="llama_index.query_engine", message=REQUIREMENT_LLAMA)
BaseQueryEngine = import_optional(mod='llama_index.indices.query.base', what='BaseQueryEngine', message=REQUIREMENT_LLAMA)
Response = import_optional(mod="llama_index.response.schema", what="Response", message=REQUIREMENT_LLAMA)

class LlamaInstrument(Instrument):

    class Default:
        MODULES = {"llama_index."}

        # Putting these inside thunk as llama_index is optional.
        CLASSES = lambda: {
            llama_index.indices.query.base.BaseQueryEngine,
            llama_index.indices.base_retriever.BaseRetriever
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
            classes=LlamaInstrument.Default.CLASSES(), # was thunk
            methods=LlamaInstrument.Default.METHODS
        )


class TruLlama(TruModel):
    """
    Wrap a llama index engine for monitoring.

    Arguments:
    - model: RetrieverQueryEngine -- the engine to wrap.
    - More args in TruModel
    - More args in LlamaModel
    - More args in WithClassInfo
    """

    class Config:
        arbitrary_types_allowed = True

    model: BaseQueryEngine

    def __init__(
        self, model: BaseQueryEngine, **kwargs
    ):

        super().update_forward_refs()

        # TruLlama specific:
        kwargs['model'] = model
        kwargs['root_class'] = Class.of_object(model)
        kwargs['instrument'] = LlamaInstrument()

        super().__init__(**kwargs)

    def query(self, *args, **kwargs) -> Response:
        res, _ = self.query_with_record(*args, **kwargs)

        return res

    def query_with_record(
        self, str_or_query_bundle
    ) -> Tuple[Response, Record]:
        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record: Sequence[RecordChainCall] = []

        ret = None
        error = None

        total_tokens = None
        total_cost = None

        start_time = None
        end_time = None

        try:
            # TODO: do this only if there is an openai model inside the chain:
            # with get_openai_callback() as cb:
            start_time = datetime.now()
            ret = self.model.query(str_or_query_bundle)
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
