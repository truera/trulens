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

OUTDATED: Output (the ... are the same as in the above chain parameter dictionary):

```json

[{...
  'chain': {...,
   '_call': {'input': {'inputs': {'question': 'hello there'}},
    'start_time': '2023-05-07 23:36:13.391052',
    'end_time': '2023-05-07 23:36:13.566101',
    'pid': 2340147,
    'tid': 2340147,
    'chain_stack': [('chain',)],
    'output': {'text': " hey. I'm here. G-g-you can tell us some things"}}},
  ...},
 {...,
  'chain': {...,
   '_call': {'input': {'inputs': {'question': 'hello there general kanobi'}},
    'start_time': '2023-05-07 23:36:13.573193',
    'end_time': '2023-05-07 23:36:13.707811',
    'pid': 2340147,
    'tid': 2340147,
    'chain_stack': [('chain',)],
    'output': {'text': ' what is the name of the game? Katsu Kato: mewt'}}},
  ...}]

  ```

- OUTDATED: Query aspects of those records via TinyDB-like queries, producing a structured
  pandas.DataFrame:

```python

    db.select(
        Record.chain.prompt.template,
        Record.chain._call.args.inputs.question,
        Record.chain._call.rets.text,
        where = Record.chain._call.output.text != None
    )

```

Output (pd.DataFrame):

```

    DATA FRAME HERE

```
"""

from collections import defaultdict
from datetime import datetime
from inspect import BoundArguments
from inspect import signature
from inspect import stack
import os
import threading as th
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import langchain

langchain.verbose = False
from langchain.chains.base import Chain
from pydantic import BaseModel
from pydantic import Field
from trulens_evalchain.tru_db import obj_id_of_obj
from trulens_evalchain.tru_db import Query
from trulens_evalchain.tru_db import Record
from trulens_evalchain.tru_db import TruDB

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

    def __init__(
        self,
        chain: Chain,
        chain_id: Optional[str] = None,
        verbose: bool = False,
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

        self._instrument(obj=self.chain, query=Record.chain)
        self.recording = False

        chain_def = self.chain_def

        # Track chain id. This will produce a name if not provided.
        self.chain_id = chain_id or obj_id_of_obj(obj=chain_def, prefix="chain")

    @property
    def chain_def(self):
        return self.dict()

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

        # def _run(self, *args: Any, callbacks: Callbacks = None, **kwargs: Any) -> str:
        # TODO (piotrm): Need to figure out what to return here.
        """Run the chain as text in, text out or multiple variables, text out."""
        """if len(self.output_keys) != 1:
            raise ValueError(
                f"`run` not supported when there is not exactly "
                f"one output key. Got {self.output_keys}."
            )

        if args and not kwargs:
            if len(args) != 1:
                raise ValueError("`run` supports only one positional argument.")
            return self(args[0], callbacks=callbacks)# [self.output_keys[0]]

        if kwargs and not args:
            return self(kwargs, callbacks=callbacks)# [self.output_keys[0]]

        if not kwargs and not args:
            raise ValueError(
                "`run` supported with either positional arguments or keyword arguments,"
                " but none were provided."
            )

        raise ValueError(
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )"""

    # langchain.chains.base.py:Chain
    # We need to override this because we defined TruChain as a Chain and the default
    # behaviour from the parent is not the same as the behaviour of the wrapped chain.
    def __call__(
        self, *args, **kwargs
    ):  #-> Dict[str, Any]: TODO(piotrm): fix type
        """
        Wrapped call to self.chain.__call__ with instrumentation.
        """

        print("Calling wrapped chain.")

        # Mark us as recording calls. Should be sufficient for non-threaded cases.
        self.recording = True

        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record = defaultdict(list)

        ret = None
        error = None

        try:
            ret = self.chain.__call__(*args, **kwargs)

        except BaseException as e:
            error = e
            print(f"WARNING: {e}")

        self.recording = False

        assert len(record) > 0, "No information recorded in call."

        ret_record = self.chain_def
        for path, calls in record.items():
            obj = TruDB._project(path=path, obj=ret_record)
            if obj is None:
                print(f"WARNING: Cannot locate {path} in chain.")
                record['_call_not_found_in_chain'] = calls
            else:
                obj.update(dict(_call=calls))

        if error is None:
            return ret, ret_record
        else:
            raise error

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
            print(
                f"WARNING: will not be able to serialize object of type {cls} because it has memory."
            )

        def safe_dict(s, json: bool = True, **kwargs: Any) -> Dict:
            """
            Return dictionary representation `s`. If `json` is set, will make sure output can be serialized.
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

                ret = dict(error=str(e))
                ret['class'] = obj.__class__.__name__
                ret['module'] = obj.__class__.__module__

                return ret

        safe_type._instrumented = prop
        new_prop = property(fget=safe_type)

        return new_prop

    def _instrument_method(self, query: Query, func: Callable):
        """
        Instrument a Chain method to capture its inputs/outputs/errors.
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
                key="record", func=TruChain.__call__
            )

            if record is None:
                return func(*args, **kwargs)

            else:
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
                    k: TruDB.dictify(v)
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

                # If there already is a call recorded at the same path, turn the calls into a list.
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

    def _instrument(self, obj, query: Query):
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
                    self._instrument_method(query=query, func=original_fun)
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
                    self._instrument(obj=v, query=query[k])

                elif isinstance(v, Sequence):
                    for i, sv in enumerate(v):
                        if isinstance(sv, Chain):
                            self._instrument(obj=sv, query=query[k][i])

                # TODO: check if we want to instrument anything not accessible through __fields__ .
        else:

            print(f"WARNING: do not know how to instrument {obj}")
