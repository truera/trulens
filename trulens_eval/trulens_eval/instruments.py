"""
# Model Instrumentation

## Designs and Choices

### Model Data

We collect model components and parameters by walking over its structure and
producing a json reprensentation with everything we deem relevant to track. The
function `util.py:jsonify` is the root of this process.

#### class/system specific

##### pydantic (langchain)

Classes inheriting `pydantic.BaseModel` come with serialization to/from json in
the form of `BaseModel.dict` and `BaseModel.parse`. We do not use the
serialization to json part of this capability as a lot of langchain components
are tripped to fail it with a "will not serialize" message. However, we use make
use of pydantic `fields` to enumerate components of an object ourselves saving
us from having to filter out irrelevant internals.

We make use of pydantic's deserialization, however, even for our own internal
structures (see `schema.py`).

##### dataclasses (no present users)

The built-in dataclasses package has similar functionality to pydantic but we
presently do not handle it as we have no use cases.

##### dataclasses_json (llama_index)

Work in progress.

##### generic python (portions of llama_index and all else)

#### TruLens-specific Data

In addition to collecting model parameters, we also collect:

- (subset of components) Model class information:

    - This allows us to deserialize some objects. Pydantic models can be
      deserialized once we know their class and fields, for example.
    - This information is also used to determine component types without having
      to deserialize them first. 
    - See `schema.py:Class` for details.

#### Tricky

#### Limitations

### Functions/Methods

Methods and functions are instrumented by overwriting choice attributes in
various classes. 

#### class/system specific

##### pydantic (langchain)

Most if not all langchain components use pydantic which imposes some
restrictions but also provides some utilities. Classes inheriting
`pydantic.BaseModel` do not allow defining new attributes but existing
attributes including those provided by pydantic itself can be overwritten (like
dict, for examople). Presently, we override methods with instrumented versions.

#### Alternatives

- `intercepts` package (see https://github.com/dlshriver/intercepts)

    Low level instrumentation of functions but is architecture and platform
    dependent with no darwin nor arm64 support as of June 07, 2023.

- `sys.setprofile` (see
  https://docs.python.org/3/library/sys.html#sys.setprofile)

    Might incur much overhead and all calls and other event types get
 
    intercepted and result in a callback.

### Tricky

- 

### Calls

The instrumented versions of functions/methods record the inputs/outputs and
some additional data (see `schema.py:RecordChainCall`). As more then one
instrumented call may take place as part of a model invokation, they are
collected and returned together in the `calls` field of `schema.py:Record`.

Calls can be connected to the components containing the called method via the
`path` field of `schema.py:RecordChainCallMethod`. This class also holds
information about the instrumented method.

#### Call Data (Arguments/Returns)

The arguments to a call and its return are converted to json using the same
tools as Model Data (see above).

#### Tricky

- The same method call with the same `path` may be recorded multiple times in a
  `Record` if the method makes use of multiple of its versions in the class
  hierarchy (i.e. an extended class calls its parents for part of its task). In
  these circumstances, the `method` field of `RecordChainCallMethod` will
  distinguish the different versions of the method.

- Thread-safety -- it is tricky to use global data to keep track of instrumented
  method calls in presence of multiple threads. For this reason we do not use
  global data and instead hide instrumenting data in the call stack frames of
  the instrumentation methods. See `util.py:get_local_in_call_stack.py`.

#### Limitations

- Threads need to be started using the utility class TP in order for
  instrumented methods called in a thread to be tracked. As we rely on call
  stack for call instrumentation we need to preserve the stack before a thread
  start which python does not do.  See `util.py:TP._thread_starter`.

"""

from datetime import datetime
from inspect import BoundArguments, signature
import os
from pprint import PrettyPrinter
import logging
from typing import Any, Callable, Dict, Sequence, Union
import threading as th

from pydantic import BaseModel
from trulens_eval.trulens_eval.schema import LangChainModel, MethodIdent, RecordChainCall, RecordChainCallMethod
from trulens_eval.trulens_eval.tru_chain import TruChain

from trulens_eval.trulens_eval.tru_db import Query

import langchain
from trulens_eval.trulens_eval.tru_feedback import Feedback
from trulens_eval.trulens_eval.util import Method, get_local_in_call_stack, jsonify, noserio

logger = logging.getLogger(__name__)

class Instrument(object):

    INSTRUMENT = "__tru_instrumented"

    CLASSES_TO_INSTRUMENT = set()

    # Instrument only methods with these names and of these classes.
    METHODS_TO_INSTRUMENT = {
        "__call__": lambda o: isinstance(o, Feedback)  # Feedback
    }

    def _instrument_tracked_method(
        self, query: Query, func: Callable, method_name: str, cls: type,
        obj: object
    ):
        """
        Instrument a method to capture its inputs/outputs/errors.
        """

        logger.debug(f"instrumenting {method_name}={func} in {query}")

        if hasattr(func, Instrument.INSTRUMENT):
            logger.debug(f"{func} is already instrumented")

            # Already instrumented. Note that this may happen under expected
            # operation when the same chain is used multiple times as part of a
            # larger chain.

            # TODO: How to consistently address calls to chains that appear more
            # than once in the wrapped chain or are called more than once.
            func = getattr(func, Instrument.INSTRUMENT)

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
                path=query, method=Method.of_method(func, obj=obj)
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

        for base in [cls] + cls.mro():
            # All of mro() may need instrumentation here if some subchains call
            # superchains, and we want to capture the intermediate steps.

            if not any(issubclass(base, c) for c in Instrument.CLASSES_TO_INSTRUMENT):
                continue

            # TODO: generalize
            if not base.__module__.startswith(
                    "langchain.") and not base.__module__.startswith("trulens"):
                continue

            for method_name in Instrument.METHODS_TO_INSTRUMENT:
                if hasattr(base, method_name):
                    check_class = Instrument.METHODS_TO_INSTRUMENT[method_name]
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

            """
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
            if hasattr(base, "dict"):# and not hasattr(base.dict, "_instrumented"):
                logger.debug(f"instrumenting {base}.dict")
                setattr(base, "dict", self._instrument_dict(cls=base, obj=obj, with_class_info=True))
            """


        # Not using chain.dict() here as that recursively converts subchains to
        # dicts but we want to traverse the instantiations here.
        if isinstance(obj, BaseModel):

            for k in obj.__fields__:
                # NOTE(piotrm): may be better to use inspect.getmembers_static .
                v = getattr(obj, k)

                if isinstance(v, str):
                    pass

                # TODO: generalize
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

class LlamaInstrument(Instrument):
    pass

class LangChainInstrument(Instrument):
    CLASSES_TO_INSTRUMENT = {
        langchain.chains.base.Chain,
        langchain.vectorstores.base.BaseRetriever,
        langchain.schema.BaseRetriever,
        langchain.llms.base.BaseLLM,
        langchain.prompts.base.BasePromptTemplate,
        langchain.schema.BaseMemory,
        langchain.schema.BaseChatMessageHistory
    }

    # Instrument only methods with these names and of these classes.
    METHODS_TO_INSTRUMENT = {
        "_call": lambda o: isinstance(o, langchain.chains.base.Chain),
        "get_relevant_documents": lambda o: True,  # VectorStoreRetriever
    }

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