"""
# Langchain instrumentation and monitoring. 

## Limitations

- If the same wrapped sub-chain is called multiple times within a single call to
  the root chain, the record of this execution will not be exact with regards to
  the path to the call information. All call dictionaries will appear in a list
  addressed by the last subchain (by order in which it is instrumented). For
  example, in a sequential chain containing two of the same chain, call records
  will be addressed to the second of the (same) chains and contain a list
  describing calls of both the first and second.

- Some chains cannot be serialized/jsonized. Sequential chain is an example.
  This is a limitation of langchain itself.

- Instrumentation relies on CPython specifics, making heavy use of the `inspect`
  module which is not expected to work with other Python implementations.

```

"""

from collections import defaultdict
from datetime import datetime
from inspect import BoundArguments
from inspect import signature
from inspect import stack
import inspect
import logging
import os
from pprint import PrettyPrinter
import threading as th
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import langchain
from langchain.callbacks import get_openai_callback
from langchain.chains.base import Chain
from pydantic import BaseModel
from pydantic import Field

from trulens_eval.schema import FeedbackMode, Method
from trulens_eval.schema import LangChainModel
from trulens_eval.schema import Record
from trulens_eval.schema import RecordChainCall
from trulens_eval.schema import RecordChainCallMethod
from trulens_eval.schema import Cost
from trulens_eval.tru_db import Query
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.tru import Tru
from trulens_eval.schema import FeedbackResult
from trulens_eval.utils.langchain import CLASSES_TO_INSTRUMENT, METHODS_TO_INSTRUMENT
from trulens_eval.util import SerialModel
from trulens_eval.util import Class
from trulens_eval.util import WithClassInfo
from trulens_eval.util import get_local_in_call_stack
from trulens_eval.util import TP, JSONPath, jsonify, noserio

logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class TruChain(LangChainModel, WithClassInfo):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    """

    class Config:
        arbitrary_types_allowed = True

    # See LangChainModel for serializable fields.

    # Feedback functions to evaluate on each record.
    feedbacks: Sequence[Feedback] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    # NOTE: Maybe move to schema.Model .
    tru: Optional[Tru] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    # NOTE: Maybe mobe to schema.Model .
    db: Optional[TruDB] = Field(exclude=True)

    def __init__(
        self,
        chain: langchain.chains.base.
        Chain,  # normally pydantic does not like positional args but this one is important
        tru: Optional[Tru] = None,
        feedbacks: Optional[Sequence[Feedback]] = None,
        feedback_mode: FeedbackMode = FeedbackMode.WITH_CHAIN_THREAD,
        **kwargs
    ):
        """
        Wrap a chain for monitoring.

        Arguments:
        
        - chain: Chain -- the chain to wrap.
        - chain_id: Optional[str] -- chain name or id. If not given, the
          name is constructed from wrapped chain parameters.
        """

        if feedbacks is not None and tru is None:
            raise ValueError("Feedback logging requires `tru` to be specified.")
        feedbacks = feedbacks or []

        if tru is not None:
            kwargs['db'] = tru.db

            if feedback_mode == FeedbackMode.NONE:
                logger.warn(
                    "`tru` is specified but `feedback_mode` is FeedbackMode.NONE. "
                    "No feedback evaluation and logging will occur."
                )
        else:

            if feedback_mode != FeedbackMode.NONE:
                logger.warn(
                    f"`feedback_mode` is {feedback_mode} but `tru` was not specified. Reverting to FeedbackMode.NONE ."
                )
                feedback_mode = FeedbackMode.NONE

        kwargs['chain'] = chain
        kwargs['tru'] = tru
        kwargs['feedbacks'] = feedbacks
        kwargs['feedback_mode'] = feedback_mode

        super().update_forward_refs()
        super().__init__(obj=self, **kwargs)
        
        if tru is not None and feedback_mode != FeedbackMode.NONE:
            logger.debug(
                "Inserting chain and feedback function definitions to db."
            )
            self.db.insert_chain(chain=self)
            for f in self.feedbacks:
                self.db.insert_feedback_definition(f)

        self._instrument_object(obj=self.chain, query=Query.Query().chain)

    # Chain requirement
    @property
    def _chain_type(self):
        return "TruChain"

    # Chain requirement
    @property
    def input_keys(self) -> List[str]:
        return self.chain.input_keys

    # Chain requirement
    @property
    def output_keys(self) -> List[str]:
        return self.chain.output_keys

    # NOTE: Input signature compatible with langchain.chains.base.Chain.__call__
    def call_with_record(self, inputs: Union[Dict[str, Any], Any], **kwargs):
        """ Run the chain and also return a record metadata object.

        Returns:
            Any: chain output
            dict: record metadata
        """
        # Mark us as recording calls. Should be sufficient for non-threaded
        # cases.
        self.recording = True

        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record: Sequence[RecordChainCall] = []

        ret = None
        error = None

        total_tokens = None
        total_cost = None

        try:
            # TODO: do this only if there is an openai model inside the chain:
            with get_openai_callback() as cb:
                ret = self.chain.__call__(inputs=inputs, **kwargs)
                total_tokens = cb.total_tokens
                total_cost = cb.total_cost

        except BaseException as e:
            error = e
            logger.error(f"Chain raised an exception: {e}")

        self.recording = False

        assert len(record) > 0, "No information recorded in call."

        ret_record_args = dict()

        inputs = self.chain.prep_inputs(inputs)

        # Figure out the content of the "inputs" arg that __call__ constructs
        # for _call so we can lookup main input and output.
        input_key = self.input_keys[0]
        output_key = self.output_keys[0]

        ret_record_args['main_input'] = inputs[input_key]
        if ret is not None:
            ret_record_args['main_output'] = ret[output_key]

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

    # langchain.chains.base.py:Chain
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Wrapped call to self.chain.__call__ with instrumentation. If you need to
        get the record, use `call_with_record` instead. 
        """

        ret, _ = self.call_with_record(*args, **kwargs)

        return ret

    def _handle_record(self, record: Record):
        """
        Write out record-related info to database if set.
        """

        if self.tru is None or self.feedback_mode is None:
            return

        record_id = self.tru.add_record(record=record)

        if len(self.feedbacks) == 0:
            return

        # Add empty (to run) feedback to db.
        if self.feedback_mode == FeedbackMode.DEFERRED:
            for f in self.feedbacks:
                self.db.insert_feedback(
                    FeedbackResult(
                        name=f.name,
                        chain_id=self.chain_id,
                        record_id=record_id,
                        feedback_definition_id=f.feedback_definition_id
                    )
                )

        elif self.feedback_mode in [FeedbackMode.WITH_CHAIN,
                                    FeedbackMode.WITH_CHAIN_THREAD]:

            feedback_results = self.tru.run_feedback_functions(
                record=record, feedback_functions=self.feedbacks, chain=self
            )

            for feedback_result in feedback_results:
                self.tru.add_feedback(feedback_result)

    def _handle_error(self, record: Record, error: Exception):
        if self.db is None:
            return

    # Chain requirement
    # TODO(piotrm): figure out whether the combination of _call and __call__ is working right.
    def _call(self, *args, **kwargs) -> Any:
        return self.chain._call(*args, **kwargs)
