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

from trulens_eval.schema import FeedbackMode, Method

from trulens_eval.schema import Record
from trulens_eval.schema import RecordChainCall
from trulens_eval.schema import RecordChainCallMethod
from trulens_eval.schema import Cost
from trulens_eval.schema import Query
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.tru import Tru
from trulens_eval.schema import FeedbackResult
from trulens_eval.tru_model import TruModel
from trulens_eval.instruments import Instrument
from trulens_eval.util import get_local_in_call_stack
from trulens_eval.util import TP, JSONPath, jsonify, noserio
import llama_index
from llama_index import query_engine

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

class LlamaInstrument(Instrument):
    
    class Default:
        MODULES = {"llama_index."}

        CLASSES = {
            query_engine.retriever_query_engine.RetrieverQueryEngine
        }

        # Instrument only methods with these names and of these classes.
        METHODS = {
            "query": lambda o: isinstance(o, query_engine.retriever_query_engine.RetrieverQueryEngine),
        }

    def __init__(self):
        super().__init__(
            root_method=TruLlama.query_with_record,
            modules=LlamaInstrument.Default.MODULES,
            classes=LlamaInstrument.Default.CLASSES,
            methods=LlamaInstrument.Default.METHODS
        )


class TruLlama(TruModel):
    """
    Wrap a langchain chain for monitoring.

    Arguments:
    - chain: Chain -- the chain to wrap.
    - More args in TruModel
    - More args in LangChainModel
    - More args in WithClassInfo
    """

    class Config:
        arbitrary_types_allowed = True

    model: query_engine.retriever_query_engine.RetrieverQueryEngine = Field(exclude=True)

    def __init__(self, **kwargs):
    
        super().update_forward_refs()

        # TruLlama specific:
        kwargs['instrument'] = LlamaInstrument()

        super().__init__(**kwargs)

        i = LlamaInstrument()
        i.instrument_object(obj=self.model, query=Query.Query().model)

    def query(self, *args, **kwargs) -> llama_index.response.schema.Response:
        res, _ = self.query_with_record(*args, **kwargs)

        return res
    
    def query_with_record(self, str_or_query_bundle) -> Tuple[llama_index.response.schema.Response, Record]:
        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record: Sequence[RecordChainCall] = []

        ret = None
        error = None

        total_tokens = None
        total_cost = None

        try:
            # TODO: do this only if there is an openai model inside the chain:
            # with get_openai_callback() as cb:
            ret = self.model.query(str_or_query_bundle)
            # total_tokens = cb.total_tokens
            # total_cost = cb.total_cost
            total_tokens = 0
            total_cost = 0

        except BaseException as e:
            error = e
            logger.error(f"Chain raised an exception: {e}")
        
        assert len(record) > 0, "No information recorded in call."

        ret_record_args = dict()

        # TODO: generalize
        ret_record_args['main_input'] = str_or_query_bundle
        if ret is not None:
            # TODO: generalize and error check
            ret_record_args['main_output'] = ret.response

        ret_record_args['main_error'] = str(error)
        ret_record_args['calls'] = record
        ret_record_args['cost'] = Cost(n_tokens=total_tokens, cost=total_cost)
        ret_record_args['chain_id'] = self.chain_id

        ret_record = Record(**ret_record_args)

        if error is not None:
            if self.feedback_mode == FeedbackMode.WITH_CHAIN:
                self._handle_error(record=ret_record, error=error)

            elif self.feedback_mode in [FeedbackMode.DEFERRED,
                                        FeedbackMode.WITH_CHAIN_THREAD]:
                TP().runlater(
                    self._handle_error, record=ret_record, error=error
                )

            raise error

        if self.feedback_mode == FeedbackMode.WITH_CHAIN:
            self._handle_record(record=ret_record)

        elif self.feedback_mode in [FeedbackMode.DEFERRED,
                                    FeedbackMode.WITH_CHAIN_THREAD]:
            TP().runlater(self._handle_record, record=ret_record)

        return ret, ret_record
