"""
# App Instrumentation

## Designs and Choices

### App Data

We collect app components and parameters by walking over its structure and
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

In addition to collecting app parameters, we also collect:

- (subset of components) App class information:

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

- langchain/llama_index callbacks. Each of these packages come with some
  callback system that lets one get various intermediate app results. The
  drawbacks is the need to handle different callback systems for each system and
  potentially missing information not exposed by them.

### Tricky

- 

### Calls

The instrumented versions of functions/methods record the inputs/outputs and
some additional data (see `schema.py:RecordAppCall`). As more then one
instrumented call may take place as part of a app invokation, they are collected
and returned together in the `calls` field of `schema.py:Record`.

Calls can be connected to the components containing the called method via the
`path` field of `schema.py:RecordAppCallMethod`. This class also holds
information about the instrumented method.

#### Call Data (Arguments/Returns)

The arguments to a call and its return are converted to json using the same
tools as App Data (see above).

#### Tricky

- The same method call with the same `path` may be recorded multiple times in a
  `Record` if the method makes use of multiple of its versions in the class
  hierarchy (i.e. an extended class calls its parents for part of its task). In
  these circumstances, the `method` field of `RecordAppCallMethod` will
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

- If the same wrapped sub-app is called multiple times within a single call to
  the root app, the record of this execution will not be exact with regards to
  the path to the call information. All call paths will address the last subapp
  (by order in which it is instrumented). For example, in a sequential app
  containing two of the same app, call records will be addressed to the second
  of the (same) apps and contain a list describing calls of both the first and
  second.

- Some apps cannot be serialized/jsonized. Sequential app is an example. This is
  a limitation of langchain itself.

- Instrumentation relies on CPython specifics, making heavy use of the `inspect`
  module which is not expected to work with other Python implementations.

#### Alternatives

- langchain/llama_index callbacks. These provide information about component
  invocations but the drawbacks are need to cover disparate callback systems and
  possibly missing information not covered.

## To Decide / To discuss

### Mirroring wrapped app behaviour and disabling instrumentation

Should our wrappers behave like the wrapped apps? Current design is like this:

```python chain = ... # some langchain chain

tru = Tru() truchain = tru.Chain(chain, ...)

plain_result = chain(...) # will not be recorded

plain_result = truchain(...) # will be recorded

plain_result, record = truchain.call_with_record(...) # will be recorded, and
you get the record too

```

The problem with the above is that "call_" part of "call_with_record" is
langchain specific and implicitly so is __call__ whose behaviour we are
replicating in TruChaib. Other wrapped apps may not implement their core
functionality in "_call" or "__call__".

Alternative #1:

```python

plain_result = chain(...) # will not be recorded

truchain = tru.Chain(chain, ...)

with truchain.record() as recorder:
    plain_result = chain(...) # will be recorded

records = recorder.records # can get records

truchain(...) # NOT SUPPORTED, use chain instead ```

Here we have the benefit of not having a special method for each app type like
call_with_record. We instead use a context to indicate that we want to collect
records and retrieve them afterwards.

### Calls: Implementation Details

Our tracking of calls uses instrumentated versions of methods to manage the
recording of inputs/outputs. The instrumented methods must distinguish
themselves from invocations of apps that are being tracked from those not being
tracked, and of those that are tracked, where in the call stack a instrumented
method invocation is. To achieve this, we rely on inspecting the python call
stack for specific frames:

- Root frame -- A tracked invocation of an app starts with one of the
  main/"root" methods such as call or query. These are the bottom of the
  relevant stack we use to manage the tracking of subsequent calls. Further
  calls to instrumented methods check for the root method in the call stack and
  retrieve the collection where data is to be recorded.
  
- Prior frame -- Each instrumented call also searches for the topmost
  instrumented call (except itself) in the stack to check its immediate caller
  (by immediate we mean only among instrumented methods) which forms the basis
  of the stack information recorded alongisde the inputs/outputs.

#### Drawbacks

- Python call stacks are implementation dependent and we don't expect to operate
  on anything other than CPython.

- Python creates a fresh empty stack for each thread. Because of this, we need
  special handling of each thread created to make sure it keeps a hold of the
  stack prior to thread creation. Right now we do this in our threading utility
  class TP but a more complete solution may be the instrumentation of
  threading.Thread class.

- We require a root method to be placed on the stack to indicate the start of
  tracking. We therefore cannot implement something like a context-manager-based
  setup of the tracking system as suggested in the "To discuss" above.

#### Alternatives

- contextvars -- langchain uses these to manage contexts such as those used for
  instrumenting/tracking LLM usage. These can be used to manage call stack
  information like we do. The drawback is that these are not threadsafe or at
  least need instrumenting thread creation. We have to do a similar thing by
  requiring threads created by our utility package which does stack management
  instead of contextvar management.

"""

from datetime import datetime
from inspect import BoundArguments
from inspect import signature
import logging
import os
from pprint import PrettyPrinter
import threading as th
from typing import Callable, Dict, Iterable, Optional, Sequence, Set

from pydantic import BaseModel

from trulens_eval.feedback import Feedback
from trulens_eval.schema import Perf
from trulens_eval.schema import Query
from trulens_eval.schema import RecordAppCall
from trulens_eval.schema import RecordAppCallMethod
from trulens_eval.util import _safe_getattr
from trulens_eval.util import get_local_in_call_stack
from trulens_eval.util import jsonify
from trulens_eval.util import Method

logger = logging.getLogger(__name__)
pp = PrettyPrinter()


class Instrument(object):
    # Attribute name to be used to flag instrumented objects/methods/others.
    INSTRUMENT = "__tru_instrumented"

    # Attribute name for marking paths that address app components.
    PATH = "__tru_path"

    class Default:
        # Default instrumentation configuration. Additional components are
        # included in subclasses of `Instrument`.

        # Modules (by full name prefix) to instrument.
        MODULES = {"trulens_eval."}

        # Classes to instrument.
        CLASSES = set()

        # Methods to instrument. Methods matching name have to pass the filter
        # to be instrumented. TODO: redesign this to be a dict with classes
        # leading to method names instead.
        METHODS = {"__call__": lambda o: isinstance(o, Feedback)}

    def to_instrument_object(self, obj: object) -> bool:
        """
        Determine whether the given object should be instrumented.
        """

        # NOTE: some classes do not support issubclass but do support
        # isinstance. It is thus preferable to do isinstance checks when we can
        # avoid issublcass checks.
        return any(isinstance(obj, cls) for cls in self.classes)

    def to_instrument_class(self, cls: type) -> bool:  # class
        """
        Determine whether the given class should be instrumented.
        """

        return any(issubclass(cls, parent) for parent in self.classes)

    def to_instrument_module(self, module_name: str) -> bool:
        """
        Determine whether a module with the given (full) name should be
        instrumented.
        """

        return any(module_name.startswith(mod2) for mod2 in self.modules)

    def __init__(
        self,
        root_method: Optional[Callable] = None,
        modules: Iterable[str] = [],
        classes: Iterable[type] = [],
        methods: Dict[str, Callable] = {},
    ):
        self.root_method = root_method

        self.modules = Instrument.Default.MODULES.union(set(modules))

        self.classes = Instrument.Default.CLASSES.union(set(classes))

        self.methods = Instrument.Default.METHODS
        self.methods.update(methods)

    def instrument_tracked_method(
        self, query: Query, func: Callable, method_name: str, cls: type,
        obj: object
    ):
        """
        Instrument a method to capture its inputs/outputs/errors.
        """

        assert self.root_method is not None, "Cannot instrument method without a `root_method`."

        if hasattr(func, Instrument.INSTRUMENT):
            logger.debug(f"\t\t\t{query}: {func} is already instrumented")

            # Already instrumented. Note that this may happen under expected
            # operation when the same chain is used multiple times as part of a
            # larger chain.

            return func

            # TODO: How to consistently address calls to chains that appear more
            # than once in the wrapped chain or are called more than once.
            # func = getattr(func, Instrument.INSTRUMENT)

        logger.debug(f"\t\t\t{query}: instrumenting {method_name}={func}")

        sig = signature(func)

        def wrapper(*args, **kwargs):
            # If not within TruChain._call, call the wrapped function without
            # any recording. This check is not perfect in threaded situations so
            # the next call stack-based lookup handles the rarer cases.

            # NOTE(piotrm): Disabling this for now as it is not thread safe.
            #if not self.recording:
            #    return func(*args, **kwargs)

            logger.debug(f"{query}: calling instrumented method {func}")

            def find_root_method(f):
                # TODO: generalize
                return id(f) == id(self.root_method.__code__)

            # Look up whether the root instrumented method was called earlier in
            # the stack and "record" variable was defined there. Will use that
            # for recording the wrapped call.
            record = get_local_in_call_stack(
                key="record", func=find_root_method
            )

            if record is None:
                logger.debug(f"{query}: no record found, not recording.")
                return func(*args, **kwargs)

            # Otherwise keep track of inputs and outputs (or exception).

            error = None
            rets = None

            def find_instrumented(f):
                return id(f) == id(wrapper.__code__)

            # If a wrapped method was called in this call stack, get the prior
            # calls from this variable. Otherwise create a new chain stack.
            stack = get_local_in_call_stack(
                key="stack", func=find_instrumented, offset=1
            ) or ()
            frame_ident = RecordAppCallMethod(
                path=query, method=Method.of_method(func, obj=obj, cls=cls)
            )
            stack = stack + (frame_ident,)

            start_time = None
            end_time = None

            try:
                # Using sig bind here so we can produce a list of key-value
                # pairs even if positional arguments were provided.
                bindings: BoundArguments = sig.bind(*args, **kwargs)
                start_time = datetime.now()
                rets = func(*bindings.args, **bindings.kwargs)
                end_time = datetime.now()

            except BaseException as e:
                end_time = datetime.now()
                error = e
                error_str = str(e)

            # Don't include self in the recorded arguments.
            nonself = {
                k: jsonify(v)
                for k, v in bindings.arguments.items()
                if k != "self"
            }

            row_args = dict(
                args=nonself,
                perf=Perf(start_time=start_time, end_time=end_time),
                pid=os.getpid(),
                tid=th.get_native_id(),
                stack=stack,
                rets=rets,
                error=error_str if error is not None else None
            )
            row = RecordAppCall(**row_args)
            record.append(row)

            if error is not None:
                raise error

            return rets

        # Indicate that the wrapper is an instrumented method so that we dont
        # further instrument it in another layer accidentally.
        setattr(wrapper, Instrument.INSTRUMENT, func)

        # Put the address of the instrumented chain in the wrapper so that we
        # don't pollute its list of fields. Note that this address may be
        # deceptive if the same subchain appears multiple times in the wrapped
        # chain.
        setattr(wrapper, Instrument.PATH, query)

        return wrapper

    def instrument_object(self, obj, query: Query, done: Set[int] = None):

        done = done or set([])

        cls = type(obj)

        logger.debug(
            f"{query}: instrumenting object at {id(obj):x} of class {cls.__name__} with mro:\n\t"
            + '\n\t'.join(map(str, cls.__mro__))
        )

        if id(obj) in done:
            logger.debug(f"\t{query}: already instrumented")
            return

        done.add(id(obj))

        # NOTE: We cannot instrument chain directly and have to instead
        # instrument its class. The pydantic BaseModel does not allow instance
        # attributes that are not fields:
        # https://github.com/pydantic/pydantic/blob/11079e7e9c458c610860a5776dc398a4764d538d/pydantic/main.py#LL370C13-L370C13
        # .

        for base in list(cls.__mro__):
            # Some top part of mro() may need instrumentation here if some
            # subchains call superchains, and we want to capture the
            # intermediate steps. On the other hand we don't want to instrument
            # the very base classes such as object:
            if not self.to_instrument_module(base.__module__):
                continue

            logger.debug(f"\t{query}: instrumenting base {base.__name__}")

            for method_name in self.methods:

                if hasattr(base, method_name):
                    check_class = self.methods[method_name]
                    if not check_class(obj):
                        continue
                    original_fun = getattr(base, method_name)

                    logger.debug(f"\t\t{query}: instrumenting {method_name}")
                    setattr(
                        base, method_name,
                        self.instrument_tracked_method(
                            query=query,
                            func=original_fun,
                            method_name=method_name,
                            cls=base,
                            obj=obj
                        )
                    )

        if isinstance(obj, BaseModel):

            for k in obj.__fields__:
                # NOTE(piotrm): may be better to use inspect.getmembers_static .
                v = getattr(obj, k)

                if isinstance(v, str):
                    pass

                elif any(type(v).__module__.startswith(module_name)
                         for module_name in self.modules):
                    self.instrument_object(obj=v, query=query[k], done=done)

                elif isinstance(v, Sequence):
                    for i, sv in enumerate(v):
                        if any(isinstance(sv, cls) for cls in self.classes):
                            self.instrument_object(
                                obj=sv, query=query[k][i], done=done
                            )

                # TODO: check if we want to instrument anything in langchain not
                # accessible through __fields__ .

        elif obj.__class__.__module__.startswith("llama_index"):
            # Some llama_index objects are using dataclasses_json but most do
            # not. Have to enumerate their contents forcefully:

            for k in dir(obj):
                if k.startswith("_") and k[1:] in dir(obj):
                    # Skip those starting with _ that also have non-_ versions.
                    continue

                sv = _safe_getattr(obj, k)

                if any(isinstance(sv, cls) for cls in self.classes):
                    self.instrument_object(obj=sv, query=query[k], done=done)

        else:
            logger.debug(
                f"{query}: Do not know how to instrument object of type {cls}."
            )
