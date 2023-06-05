"""
# Langchain instrumentation and monitoring. 

## Limitations

- Uncertain thread safety.

- If the same wrapped sub-chain is called multiple times within a single call to
  the root chain, the record of this execution will not be exact with regards to
  the path to the call information. All call dictionaries will appear in a list
  addressed by the last subchain (by order in which it is instrumented). For
  example, in a sequential chain containing two of the same chain, call records
  will be addressed to the second of the (same) chains and contain a list
  describing calls of both the first and second.

- Some chains cannot be serialized/jsonized. Sequential chain is an example.
  This is a limitation of langchain itself.

## Basic Usage

- Wrap an existing chain:

```python

    tc = TruChain(t.llm_chain)

```

- Retrieve the parameters of the wrapped chain:

```python

    tc.chain

```

Output:

```json

{'memory': None,
 'verbose': False,
 'chain': {'memory': None,
  'verbose': True,
  'prompt': {'input_variables': ['question'],
   'output_parser': None,
   'partial_variables': {},
   'template': 'Q: {question} A:',
   'template_format': 'f-string',
   'validate_template': True,
   '_type': 'prompt'},
  'llm': {'model_id': 'gpt2',
   'model_kwargs': None,
   '_type': 'huggingface_pipeline'},
  'output_key': 'text',
  '_type': 'llm_chain'},
 '_type': 'TruChain'}
 
 ```

- Make calls like you would to the wrapped chain.

```python

    rec1: dict = tc("hello there")
    rec2: dict = tc("hello there general kanobi")

```

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

import langchain
from langchain.callbacks import get_openai_callback
from langchain.chains.base import Chain
from pydantic import BaseModel
from pydantic import Field

from trulens_eval.schema import FeedbackMode, Method, MethodIdent
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
from trulens_eval.util import get_local_in_call_stack
from trulens_eval.util import TP, JSONPath, jsonify, noserio

logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class TruChain(LangChainModel):
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

        kwargs['tru'] = tru
        kwargs['feedbacks'] = feedbacks
        kwargs['feedback_mode'] = feedback_mode

        super().__init__(**kwargs)

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

            results = self.tru.run_feedback_functions(
                record=record, feedback_functions=self.feedbacks, chain=self
            )

            for result in results:
                self.tru.add_feedback(result)

    def _handle_error(self, record: Record, error: Exception):
        if self.db is None:
            return

    # Chain requirement
    # TODO(piotrm): figure out whether the combination of _call and __call__ is working right.
    def _call(self, *args, **kwargs) -> Any:
        return self.chain._call(*args, **kwargs)

    def _instrument_dict(self, cls, obj: Any):
        """
        Replacement for langchain's dict method to one that does not fail under
        non-serialization situations.
        """

        if hasattr(obj, "memory"):
            if obj.memory is not None:
                # logger.warn(
                #     f"Will not be able to serialize object of type {cls} because it has memory."
                # )
                pass

        def safe_dict(s, json: bool = True, **kwargs: Any) -> Dict:
            """
            Return dictionary representation `s`. If `json` is set, will make
            sure output can be serialized.
            """

            # if s.memory is not None:
            # continue anyway
            # raise ValueError("Saving of memory is not yet supported.")

            sup = super(cls, s)
            if hasattr(sup, "dict"):
                _dict = super(cls, s).dict(**kwargs)
            else:
                _dict = {"_base_type": cls.__name__}
            # _dict = cls.dict(s, **kwargs)
            # _dict["_type"] = s._chain_type

            # TODO: json

            return _dict

        safe_dict._instrumented = getattr(cls, "dict")

        return safe_dict

    def _instrument_type_method(self, obj, prop):
        """
        Instrument the Langchain class's method _*_type which is presently used
        to control chain saving. Override the exception behaviour. Note that
        _chain_type is defined as a property in langchain.
        """

        # Properties doesn't let us new define new attributes like "_instrument"
        # so we put it on fget instead.
        if hasattr(prop.fget, "_instrumented"):
            prop = prop.fget._instrumented

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

    def _instrument_tracked_method(
        self, query: Query, func: Callable, method_name: str, cls: type,
        obj: object
    ):
        """
        Instrument a method to capture its inputs/outputs/errors.
        """

        logger.debug(f"instrumenting {method_name}={func} in {query}")

        if hasattr(func, "_instrumented"):
            logger.debug(f"{func} is already instrumented")

            # Already instrumented. Note that this may happen under expected
            # operation when the same chain is used multiple times as part of a
            # larger chain.

            # TODO: How to consistently address calls to chains that appear more
            # than once in the wrapped chain or are called more than once.
            func = func._instrumented

        sig = signature(func)

        def wrapper(*args, **kwargs):
            # If not within TruChain._call, call the wrapped function without
            # any recording. This check is not perfect in threaded situations so
            # the next call stack-based lookup handles the rarer cases.

            # NOTE(piotrm): Disabling this for now as it is not thread safe.
            #if not self.recording:
            #    return func(*args, **kwargs)

            logger.debug(f"Calling instrumented method {func} on {query}")

            def find_call_with_record(f):
                return id(f) == id(TruChain.call_with_record.__code__)

            # Look up whether TruChain._call was called earlier in the stack and
            # "record" variable was defined there. Will use that for recording
            # the wrapped call.
            record = get_local_in_call_stack(
                key="record", func=find_call_with_record
            )

            if record is None:
                logger.debug("No record found, not recording.")
                return func(*args, **kwargs)

            # Otherwise keep track of inputs and outputs (or exception).

            error = None
            rets = None

            start_time = datetime.now()

            def find_instrumented(f):
                return id(f) == id(wrapper.__code__)
                # return hasattr(f, "_instrumented")

            # If a wrapped method was called in this call stack, get the prior
            # calls from this variable. Otherwise create a new chain stack.
            chain_stack = get_local_in_call_stack(
                key="chain_stack", func=find_instrumented, offset=1
            ) or ()
            frame_ident = RecordChainCallMethod(
                path=query, method=MethodIdent.of_method(func, obj=obj)
            )
            chain_stack = chain_stack + (frame_ident,)

            try:
                # Using sig bind here so we can produce a list of key-value
                # pairs even if positional arguments were provided.
                bindings: BoundArguments = sig.bind(*args, **kwargs)
                rets = func(*bindings.args, **bindings.kwargs)

            except BaseException as e:
                error = e
                error_str = str(e)

            end_time = datetime.now()

            # Don't include self in the recorded arguments.
            nonself = {
                k: jsonify(v)
                for k, v in bindings.arguments.items()
                if k != "self"
            }

            row_args = dict(
                args=nonself,
                start_time=start_time,
                end_time=end_time,
                pid=os.getpid(),
                tid=th.get_native_id(),
                chain_stack=chain_stack,
                rets=rets,
                error=error_str if error is not None else None
            )

            row = RecordChainCall(**row_args)
            record.append(row)

            if error is not None:
                raise error

            return rets

        wrapper._instrumented = func

        # Put the address of the instrumented chain in the wrapper so that we
        # don't pollute its list of fields. Note that this address may be
        # deceptive if the same subchain appears multiple times in the wrapped
        # chain.
        wrapper._query = query

        return wrapper

    def _instrument_object(self, obj, query: Query):

        cls = type(obj)

        logger.debug(
            f"instrumenting {query} {cls.__name__}, bases={cls.__bases__}"
        )

        # NOTE: We cannot instrument chain directly and have to instead
        # instrument its class. The pydantic BaseModel does not allow instance
        # attributes that are not fields:
        # https://github.com/pydantic/pydantic/blob/11079e7e9c458c610860a5776dc398a4764d538d/pydantic/main.py#LL370C13-L370C13
        # .

        # Instrument only methods with these names and of these classes.
        methods_to_instrument = {
            "_call": lambda o: isinstance(o, langchain.chains.base.Chain),
            "get_relevant_documents": lambda o: True,  # VectorStoreRetriever
            "__call__":
                lambda o: isinstance(o, Feedback)  # Feedback
        }

        for base in [cls] + cls.mro():
            # All of mro() may need instrumentation here if some subchains call
            # superchains, and we want to capture the intermediate steps.

            if not base.__module__.startswith(
                    "langchain.") and not base.__module__.startswith("trulens"):
                continue

            for method_name in methods_to_instrument:
                if hasattr(base, method_name):
                    check_class = methods_to_instrument[method_name]
                    if not check_class(obj):
                        continue

                    original_fun = getattr(base, method_name)

                    logger.debug(f"instrumenting {base}.{method_name}")

                    setattr(
                        base, method_name,
                        self._instrument_tracked_method(
                            query=query,
                            func=original_fun,
                            method_name=method_name,
                            cls=base,
                            obj=obj
                        )
                    )

            # Instrument special langchain methods that may cause serialization
            # failures.
            if hasattr(base, "_chain_type"):
                logger.debug(f"instrumenting {base}._chain_type")

                prop = getattr(base, "_chain_type")
                setattr(
                    base, "_chain_type",
                    self._instrument_type_method(obj=obj, prop=prop)
                )

            if hasattr(base, "_prompt_type"):
                logger.debug(f"instrumenting {base}._chain_prompt")

                prop = getattr(base, "_prompt_type")
                setattr(
                    base, "_prompt_type",
                    self._instrument_type_method(obj=obj, prop=prop)
                )

            # Instrument a pydantic.BaseModel method that may cause
            # serialization failures.
            if hasattr(base, "dict"):
                logger.debug(f"instrumenting {base}.dict")

                setattr(base, "dict", self._instrument_dict(cls=base, obj=obj))

        # Not using chain.dict() here as that recursively converts subchains to
        # dicts but we want to traverse the instantiations here.
        if isinstance(obj, BaseModel):

            for k in obj.__fields__:
                # NOTE(piotrm): may be better to use inspect.getmembers_static .
                v = getattr(obj, k)

                if isinstance(v, str):
                    pass

                elif type(v).__module__.startswith("langchain.") or type(
                        v).__module__.startswith("trulens"):
                    self._instrument_object(obj=v, query=query[k])

                elif isinstance(v, Sequence):
                    for i, sv in enumerate(v):
                        if isinstance(sv, langchain.chains.base.Chain):
                            self._instrument_object(obj=sv, query=query[k][i])

                # TODO: check if we want to instrument anything not accessible through __fields__ .
        else:
            logger.debug(
                f"Do not know how to instrument object {str(obj)[:32]} of type {cls}."
            )
