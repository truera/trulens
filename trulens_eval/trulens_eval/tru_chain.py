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
import threading as th
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import langchain
from langchain.callbacks import get_openai_callback
from langchain.chains.base import Chain
from pydantic import BaseModel
from pydantic import Field

from trulens_eval.tru_db import JSON, noserio
from trulens_eval.tru_db import obj_id_of_obj
from trulens_eval.tru_db import Query
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import TP

langchain.verbose = False

# Addresses of chains or their contents. This is used to refer chains/parameters
# even in cases where the live object is not in memory (i.e. on some remote
# app).
Path = Tuple[Union[str, int], ...]

# Records of a chain run are dictionaries with these keys:
#
# - 'args': Dict[str, Any] -- chain __call__ input args.
# - 'rets': Dict[str, Any] -- chain __call__ return dict for calls that succeed.
# - 'error': str -- exception text if not successful.
# - 'start_time': datetime
# - 'end_time': datetime -- runtime info.
# - 'pid': int -- process id for debugging multiprocessing.
# - 'tid': int -- thread id for debuggin threading.
# - 'chain_stack': List[Path] -- call stack of chain runs. Elements address
#   chains.


class TruChain(Chain):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    """

    # The wrapped/instrumented chain.
    chain: Chain = None

    # Chain name/id. Will be a hash of chain definition/configuration if not provided.
    chain_id: Optional[str] = None

    # Flag of whether the chain is currently recording records. This is set
    # automatically but is imperfect in threaded situations. The second check
    # for recording is based on the call stack, see _call.
    recording: Optional[bool] = Field(exclude=True)

    # Feedback functions to evaluate on each record.
    feedbacks: Optional[Sequence[Feedback]] = Field(exclude=True)

    # Feedback evaluation mode.
    # - "withchain" - Try to run feedback functions immediately and before chain
    #   returns a record.
    # - "withchainthread" - Try to run feedback functions in the same process as
    #   the chain but after it produces a record.
    # - "deferred" - Evaluate later via the process started by
    #   `tru.start_deferred_feedback_evaluator`.
    # - None - No evaluation will happen even if feedback functions are specified.

    # NOTE: Custom feedback functions cannot be run deferred and will be run as
    # if "withchainthread" was set.
    feedback_mode: Optional[str] = "withchainthread"

    # Database interfaces for models/records/feedbacks.
    tru: Optional['Tru'] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    db: Optional[TruDB] = Field(exclude=True)

    def __init__(
        self,
        chain: Chain,
        chain_id: Optional[str] = None,
        verbose: bool = False,
        feedbacks: Optional[Sequence[Feedback]] = None,
        feedback_mode: Optional[str] = "withchainthread",
        tru: Optional['Tru'] = None
    ):
        """
        Wrap a chain for monitoring.

        Arguments:
        
        - chain: Chain -- the chain to wrap.
        - chain_id: Optional[str] -- chain name or id. If not given, the
          name is constructed from wrapped chain parameters.
        """

        Chain.__init__(self, verbose=verbose)

        self.chain = chain

        self._instrument_object(obj=self.chain, query=Query().chain)
        self.recording = False

        chain_def = self.json

        # Track chain id. This will produce a name if not provided.
        self.chain_id = chain_id or obj_id_of_obj(obj=chain_def, prefix="chain")

        if feedbacks is not None and tru is None:
            raise ValueError("Feedback logging requires `tru` to be specified.")

        self.feedbacks = feedbacks or []

        assert feedback_mode in [
            'withchain', 'withchainthread', 'deferred', None
        ], "`feedback_mode` must be one of 'withchain', 'withchainthread', 'deferred', or None."
        self.feedback_mode = feedback_mode

        if tru is not None:
            self.db = tru.db

            if feedback_mode is None:
                logging.warn(
                    "`tru` is specified but `feedback_mode` is None. "
                    "No feedback evaluation and logging will occur."
                )
        else:
            if feedback_mode is not None:
                logging.warn(
                    f"`feedback_mode` is {feedback_mode} but `tru` was not specified. Reverting to None."

                )
                self.feedback_mode = None
                feedback_mode = None
                # Import here to avoid circular imports.
                # from trulens_eval import Tru
                # tru = Tru()

        self.tru = tru

        if tru is not None and feedback_mode is not None:
            logging.debug(
                "Inserting chain and feedback function definitions to db."
            )
            self.db.insert_chain(chain_id=self.chain_id, chain_json=self.json)
            for f in self.feedbacks:
                self.db.insert_feedback_def(f.json)

    @property
    def json(self):
        temp = TruDB.jsonify(self)  # not using self.dict()
        # Need these to run feedback functions when they don't specify their
        # inputs exactly.

        temp['input_keys'] = self.input_keys
        temp['output_keys'] = self.output_keys

        return temp

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

    def call_with_record(self, *args, **kwargs):
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
        record = defaultdict(list)

        ret = None
        error = None

        total_tokens = None
        total_cost = None

        try:
            # TODO: do this only if there is an openai model inside the chain:
            with get_openai_callback() as cb:
                ret = self.chain.__call__(*args, **kwargs)
                total_tokens = cb.total_tokens
                total_cost = cb.total_cost

        except BaseException as e:
            error = e
            logging.error(f"Chain raised an exception: {e}")

        self.recording = False

        assert len(record) > 0, "No information recorded in call."

        ret_record = dict()
        chain_json = self.json

        for path, calls in record.items():
            obj = TruDB._project(path=path, obj=chain_json)

            if obj is None:
                logging.warn(f"Cannot locate {path} in chain.")

            ret_record = TruDB._set_in_json(
                path=path, in_json=ret_record, val={"_call": calls}
            )

        ret_record['_cost'] = dict(
            total_tokens=total_tokens, total_cost=total_cost
        )
        ret_record['chain_id'] = self.chain_id

        if error is not None:

            if self.feedback_mode == "withchain":            
                self._handle_error(record_json=ret_record, error=error)

            elif self.feedback_mode in ["deferred", "withchainthread"]:
                TP().runlater(
                    self._handle_error, record_json=ret_record, error=error
                )

            raise error

        if self.feedback_mode == "withchain":
            self._handle_record(record_json=ret_record)

        elif self.feedback_mode in ["deferred", "withchainthread"]:
            TP().runlater(self._handle_record, record_json=ret_record)

        return ret, ret_record

    # langchain.chains.base.py:Chain
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Wrapped call to self.chain.__call__ with instrumentation. If you need to
        get the record, use `call_with_record` instead. 
        """

        ret, record = self.call_with_record(*args, **kwargs)

        return ret

    def _handle_record(self, record_json: JSON):
        """
        Write out record-related info to database if set.
        """

        if self.tru is None or self.feedback_mode is None:
            return

        main_input = record_json['chain']['_call']['args']['inputs'][
            self.input_keys[0]]
        main_output = record_json['chain']['_call']['rets'][self.output_keys[0]]

        record_id = self.tru.add_record(
            prompt=main_input,
            response=main_output,
            record_json=record_json,
            tags='dev',  # TODO: generalize
            total_tokens=record_json['_cost']['total_tokens'],
            total_cost=record_json['_cost']['total_cost']
        )

        if len(self.feedbacks) == 0:
            return

        # Add empty (to run) feedback to db.
        if self.feedback_mode == "deferred":
            for f in self.feedbacks:
                feedback_id = f.feedback_id
                self.db.insert_feedback(record_id, feedback_id)

        elif self.feedback_mode in ["withchain", "withchainthread"]:
            
            results = self.tru.run_feedback_functions(
                record_json=record_json,
                feedback_functions=self.feedbacks,
                chain_json=self.json
            )

            for result_json in results:
                self.tru.add_feedback(result_json)

    def _handle_error(self, record, error):
        if self.db is None:
            return

        pass

    # Chain requirement
    # TODO(piotrm): figure out whether the combination of _call and __call__ is working right.
    def _call(self, *args, **kwargs) -> Any:
        return self.chain._call(*args, **kwargs)

    def _get_local_in_call_stack(
        self, key: str, func: Callable, offset: int = 1
    ) -> Optional[Any]:
        """
        Get the value of the local variable named `key` in the stack at the
        nearest frame executing `func`. Returns None if `func` is not in call
        stack. Raises RuntimeError if `func` is in call stack but does not have
        `key` in its locals.
        """

        for fi in stack()[offset + 1:]:  # + 1 to skip this method itself
            if id(fi.frame.f_code) == id(func.__code__):
                locs = fi.frame.f_locals
                if key in locs:
                    return locs[key]
                else:
                    raise RuntimeError(f"No local named {key} in {func} found.")

        return None

    def _instrument_dict(self, cls, obj: Any):
        """
        Replacement for langchain's dict method to one that does not fail under
        non-serialization situations.
        """

        if obj.memory is not None:

            # logging.warn(
            #     f"Will not be able to serialize object of type {cls} because it has memory."
            # )

            pass

        def safe_dict(s, json: bool = True, **kwargs: Any) -> Dict:
            """
            Return dictionary representation `s`. If `json` is set, will make
            sure output can be serialized.
            """

            #if s.memory is not None:
            # continue anyway
            # raise ValueError("Saving of memory is not yet supported.")

            _dict = super(cls, s).dict(**kwargs)
            _dict["_type"] = s._chain_type

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

    def _instrument_call(self, query: Query, func: Callable):
        """
        Instrument a Chain.__call__ method to capture its inputs/outputs/errors.
        """

        if hasattr(func, "_instrumented"):
            if self.verbose:
                print(f"{func} is already instrumented")

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

            # Look up whether TruChain._call was called earlier in the stack and
            # "record" variable was defined there. Will use that for recording
            # the wrapped call.
            record = self._get_local_in_call_stack(
                key="record", func=TruChain.call_with_record
            )

            if record is None:
                return func(*args, **kwargs)

            # Otherwise keep track of inputs and outputs (or exception).

            error = None
            ret = None

            start_time = datetime.now()

            chain_stack = self._get_local_in_call_stack(
                key="chain_stack", func=wrapper, offset=1
            ) or []
            chain_stack = chain_stack + [query._path]

            try:
                # Using sig bind here so we can produce a list of key-value
                # pairs even if positional arguments were provided.
                bindings: BoundArguments = sig.bind(*args, **kwargs)
                ret = func(*bindings.args, **bindings.kwargs)

            except BaseException as e:
                error = e

            end_time = datetime.now()

            # Don't include self in the recorded arguments.
            nonself = {
                k: TruDB.jsonify(v)
                for k, v in bindings.arguments.items()
                if k != "self"
            }
            row_args = dict(
                args=nonself,
                start_time=str(start_time),
                end_time=str(end_time),
                pid=os.getpid(),
                tid=th.get_native_id(),
                chain_stack=chain_stack
            )

            if error is not None:
                row_args['error'] = error
            else:
                row_args['rets'] = ret

            # If there already is a call recorded at the same path, turn the
            # calls into a list.
            if query._path in record:
                existing_call = record[query._path]
                if isinstance(existing_call, dict):
                    record[query._path] = [existing_call, row_args]
                else:
                    record[query._path].append(row_args)
            else:
                # Otherwise record just the one call not inside a list.
                record[query._path] = row_args

            if error is not None:
                raise error

            return ret

        wrapper._instrumented = func

        # Put the address of the instrumented chain in the wrapper so that we
        # don't pollute its list of fields. Note that this address may be
        # deceptive if the same subchain appears multiple times in the wrapped
        # chain.
        wrapper._query = query

        return wrapper

    def _instrument_object(self, obj, query: Query):
        if self.verbose:
            print(f"instrumenting {query._path} {obj.__class__.__name__}")

        cls = obj.__class__

        # NOTE: We cannot instrument chain directly and have to instead
        # instrument its class. The pydantic BaseModel does not allow instance
        # attributes that are not fields:
        # https://github.com/pydantic/pydantic/blob/11079e7e9c458c610860a5776dc398a4764d538d/pydantic/main.py#LL370C13-L370C13
        # .
        for base in cls.mro():
            # All of mro() may need instrumentation here if some subchains call
            # superchains, and we want to capture the intermediate steps.

            if not base.__module__.startswith("langchain."):
                continue

            if hasattr(base, "_call"):
                original_fun = getattr(base, "_call")

                if self.verbose:
                    print(f"instrumenting {base}._call")

                setattr(
                    base, "_call",
                    self._instrument_call(query=query, func=original_fun)
                )

            if hasattr(base, "_chain_type"):
                if self.verbose:
                    print(f"instrumenting {base}._chain_type")

                prop = getattr(base, "_chain_type")
                setattr(
                    base, "_chain_type",
                    self._instrument_type_method(obj=obj, prop=prop)
                )

            if hasattr(base, "_prompt_type"):
                if self.verbose:
                    print(f"instrumenting {base}._chain_prompt")

                prop = getattr(base, "_prompt_type")
                setattr(
                    base, "_prompt_type",
                    self._instrument_type_method(obj=obj, prop=prop)
                )

            if isinstance(obj, Chain):
                if self.verbose:
                    print(f"instrumenting {base}.dict")

                setattr(base, "dict", self._instrument_dict(cls=base, obj=obj))

        # Not using chain.dict() here as that recursively converts subchains to
        # dicts but we want to traverse the instantiations here.
        if isinstance(obj, BaseModel):

            for k in obj.__fields__:
                # NOTE(piotrm): may be better to use inspect.getmembers_static .
                v = getattr(obj, k)

                if isinstance(v, str):
                    pass

                elif v.__class__.__module__.startswith("langchain."):
                    self._instrument_object(obj=v, query=query[k])

                elif isinstance(v, Sequence):
                    for i, sv in enumerate(v):
                        if isinstance(sv, Chain):
                            self._instrument_object(obj=sv, query=query[k][i])

                # TODO: check if we want to instrument anything not accessible through __fields__ .
        else:
            logging.debug(
                f"Do not know how to instrument object {str(obj)[:32]} of type {type(obj)}."
            )
