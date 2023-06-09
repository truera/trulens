"""
# Langchain instrumentation and monitoring. 
```

"""

import logging

from pprint import PrettyPrinter

from typing import Any, Dict, List, Optional, Sequence, Union

import langchain
from langchain.callbacks import get_openai_callback
from pydantic import Field

from trulens_eval.tru_model import FeedbackMode
from trulens_eval.schema import Record
from trulens_eval.schema import RecordChainCall
from trulens_eval.schema import Cost
from trulens_eval.schema import Query
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.tru import Tru
from trulens_eval.schema import FeedbackResult

from trulens_eval.tru_model import TruModel
from trulens_eval.instruments import Instrument
from trulens_eval.util import jsonify, noserio
from trulens_eval.util import WithClassInfo

from trulens_eval.util import TP

logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class LangChainInstrument(Instrument):
    class Default:
        MODULES = {"langchain."}

        CLASSES = {
            langchain.chains.base.Chain,
            langchain.vectorstores.base.BaseRetriever,
            langchain.schema.BaseRetriever,
            langchain.llms.base.BaseLLM,
            langchain.prompts.base.BasePromptTemplate,
            langchain.schema.BaseMemory,
            langchain.schema.BaseChatMessageHistory
        }

        # Instrument only methods with these names and of these classes.
        METHODS = {
            "_call": lambda o: isinstance(o, langchain.chains.base.Chain),
            "get_relevant_documents": lambda o: True,  # VectorStoreRetriever
        }

    def __init__(self):
        super().__init__(
            root_method=TruChain.call_with_record,
            modules=LangChainInstrument.Default.MODULES,
            classes=LangChainInstrument.Default.CLASSES,
            methods=LangChainInstrument.Default.METHODS
        )

    def _instrument_dict(self, cls, obj: Any, with_class_info: bool = False):
        """
        Replacement for langchain's dict method to one that does not fail under
        non-serialization situations.
        """

        return jsonify

    def _instrument_type_method(self, obj, prop):
        """
        Instrument the Langchain class's method _*_type which is presently used
        to control chain saving. Override the exception behaviour. Note that
        _chain_type is defined as a property in langchain.
        """

        # Properties doesn't let us new define new attributes like "_instrument"
        # so we put it on fget instead.
        if hasattr(prop.fget, Instrument.INSTRUMENT):
            prop = getattr(prop.fget, Instrument.INSTRUMENT)

        def safe_type(s) -> Union[str, Dict]:
            # self should be chain
            try:
                ret = prop.fget(s)
                return ret

            except NotImplementedError as e:

                return noserio(obj, error=f"{e.__class__.__name__}='{str(e)}'")

        safe_type._instrumented = prop
        new_prop = property(fget=safe_type)

        return new_prop
    

class TruChain(TruModel):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    """

    model: langchain.chains.base.Chain

     # Normally pydantic does not like positional args but chain here is
     # important enough to make an exception.
    def __init__(
        self,
        model: langchain.chains.base.Chain,
        **kwargs
    ):
        """
        Wrap a langchain chain for monitoring.

        Arguments:
        - chain: Chain -- the chain to wrap.
        - More args in TruModel
        - More args in LangChainModel
        - More args in WithClassInfo
        """

        super().update_forward_refs()

        # TruChain specific:
        kwargs['model'] = model
        kwargs['instrument'] = LangChainInstrument()

        super().__init__(**kwargs)

    # Chain requirement
    @property
    def _chain_type(self):
        return "TruChain"

    # Chain requirement
    @property
    def input_keys(self) -> List[str]:
        return self.model.input_keys

    # Chain requirement
    @property
    def output_keys(self) -> List[str]:
        return self.model.output_keys

    # NOTE: Input signature compatible with langchain.chains.base.Chain.__call__
    def call_with_record(self, inputs: Union[Dict[str, Any], Any], **kwargs):
        """ Run the chain and also return a record metadata object.

        Returns:
            Any: chain output
            dict: record metadata
        """
        
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
                ret = self.model.__call__(inputs=inputs, **kwargs)
                total_tokens = cb.total_tokens
                total_cost = cb.total_cost

        except BaseException as e:
            error = e
            logger.error(f"Chain raised an exception: {e}")
        
        assert len(record) > 0, "No information recorded in call."

        ret_record_args = dict()

        inputs = self.model.prep_inputs(inputs)

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
        Wrapped call to self.model.__call__ with instrumentation. If you need to
        get the record, use `call_with_record` instead. 
        """

        ret, _ = self.call_with_record(*args, **kwargs)

        return ret

    # Chain requirement
    # TODO(piotrm): figure out whether the combination of _call and __call__ is working right.
    def _call(self, *args, **kwargs) -> Any:
        return self.model._call(*args, **kwargs)
