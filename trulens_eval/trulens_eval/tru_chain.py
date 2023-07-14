"""
# Langchain instrumentation and monitoring.
"""

from datetime import datetime
import logging
from pprint import PrettyPrinter
from typing import Any, ClassVar, Dict, List, Sequence, Union

from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.provider_apis import Endpoint
from trulens_eval.schema import Cost
from trulens_eval.schema import RecordAppCall
from trulens_eval.util import Class
from trulens_eval.util import FunctionOrMethod
from trulens_eval.util import jsonify
from trulens_eval.util import noserio
from trulens_eval.util import OptionalImports
from trulens_eval.util import REQUIREMENT_LANGCHAIN

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

with OptionalImports(message=REQUIREMENT_LANGCHAIN):
    import langchain
    from langchain.callbacks import get_openai_callback
    from langchain.chains.base import Chain


class LangChainInstrument(Instrument):

    class Default:
        MODULES = {"langchain."}

        # Thunk because langchain is optional.
        CLASSES = lambda: {
            langchain.chains.base.Chain, langchain.vectorstores.base.
            BaseRetriever, langchain.schema.BaseRetriever, langchain.llms.base.
            BaseLLM, langchain.prompts.base.BasePromptTemplate, langchain.schema
            .BaseMemory, langchain.schema.BaseChatMessageHistory
        }

        # Instrument only methods with these names and of these classes.
        METHODS = {
            "_call": lambda o: isinstance(o, langchain.chains.base.Chain),
            # "get_relevant_documents": lambda o: True,  # VectorStoreRetriever
            "_get_relevant_documents":
                lambda o: True,  # VectorStoreRetriever, langchain >= 0.230
        }

    def __init__(self):
        super().__init__(
            root_method=TruChain.call_with_record,
            modules=LangChainInstrument.Default.MODULES,
            classes=LangChainInstrument.Default.CLASSES(),
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


class TruChain(App):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    """

    app: Chain

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruChain._call),
        const=True
    )

    # Normally pydantic does not like positional args but chain here is
    # important enough to make an exception.
    def __init__(self, app: Chain, **kwargs):
        """
        Wrap a langchain chain for monitoring.

        Arguments:
        - app: Chain -- the chain to wrap.
        - More args in App
        - More args in AppDefinition
        - More args in WithClassInfo
        """

        super().update_forward_refs()

        # TruChain specific:
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = LangChainInstrument()

        super().__init__(**kwargs)

    # Chain requirement
    @property
    def _chain_type(self):
        return "TruChain"

    # Chain requirement
    @property
    def input_keys(self) -> List[str]:
        return self.app.input_keys

    # Chain requirement
    @property
    def output_keys(self) -> List[str]:
        return self.app.output_keys

    def __getattr__(self, __name: str) -> Any:
        # A message for cases where a user calls something that the wrapped
        # chain has but we do not wrap yet.

        if hasattr(self.app, __name):
            return RuntimeError(
                f"TruChain has no attribute {__name} but the wrapped app ({type(self.app)}) does. ",
                f"If you are calling a {type(self.app)} method, retrieve it from that app instead of from `TruChain`. "
                f"TruChain only wraps the the Chain.__call__ and Chain._call methods presently."
            )
        else:
            raise RuntimeError(f"TruChain has no attribute named {__name}.")

    # NOTE: Input signature compatible with langchain.chains.base.Chain.__call__
    def call_with_record(self, inputs: Union[Dict[str, Any], Any], **kwargs):
        """ Run the chain and also return a record metadata object.

        Returns:
            Any: chain output
            dict: record metadata
        """

        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record: Sequence[RecordAppCall] = []

        ret = None
        error = None

        cost: Cost = Cost()

        start_time = None
        end_time = None

        try:
            # TODO: do this only if there is an openai model inside the chain:
            with get_openai_callback() as cb:
                start_time = datetime.now()
                ret, cost = Endpoint.track_all_costs_tally(
                    lambda: self.app.__call__(inputs=inputs, **kwargs)
                )
                end_time = datetime.now()

        except BaseException as e:
            end_time = datetime.now()
            error = e
            logger.error(f"App raised an exception: {e}")

        assert len(record) > 0, "No information recorded in call."

        ret_record_args = dict()

        inputs = self.app.prep_inputs(inputs)

        # Figure out the content of the "inputs" arg that __call__ constructs
        # for _call so we can lookup main input and output.
        input_key = self.input_keys[0]
        output_key = self.output_keys[0]

        ret_record_args['main_input'] = jsonify(inputs[input_key])

        if ret is not None:
            ret_record_args['main_output'] = jsonify(ret[output_key])

        if error is not None:
            ret_record_args['main_error'] = jsonify(error)

        ret_record = self._post_record(
            ret_record_args, error, cost, start_time, end_time, record
        )

        return ret, ret_record

    # langchain.chains.base.py:Chain requirement:
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Wrapped call to self.app.__call__ with instrumentation. If you need to
        get the record, use `call_with_record` instead. 
        """

        ret, _ = self.call_with_record(*args, **kwargs)

        return ret

    # langchain.chains.base.py:Chain requirement:
    def _call(self, *args, **kwargs) -> Any:
        # TODO(piotrm): figure out whether the combination of _call and __call__ is
        # working right.

        # TODO(piotrm): potentially remove this. We don't want to be
        # wrapping/passing through all of the methods that a langchain Chain
        # supports.

        return self.app._call(*args, **kwargs)
