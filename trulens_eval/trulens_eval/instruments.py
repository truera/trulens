"""
# App Instrumentation

## Designs and Choices

### App Data

We collect app components and parameters by walking over its structure and
producing a json reprensentation with everything we deem relevant to track. The
function `utils/json.py:jsonify` is the root of this process.

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

The built-in dataclasses package has similar functionality to pydantic. We
use/serialize them using their field information.

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
    - See `utils/pyschema.py:Class` for details.

### Functions/Methods

Methods and functions are instrumented by overwriting choice attributes in
various classes. 

#### class/system specific

##### pydantic (langchain)

Most if not all langchain components use pydantic which imposes some
restrictions but also provides some utilities. Classes inheriting
`pydantic.BaseModel` do not allow defining new attributes but existing
attributes including those provided by pydantic itself can be overwritten (like
dict, for example). Presently, we override methods with instrumented versions.

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

### Calls

The instrumented versions of functions/methods record the inputs/outputs and
some additional data (see `schema.py:RecordAppCall`). As more than one
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
  the instrumentation methods. See
  `utils/python.py:get_all_local_in_call_stack`.

#### Threads

Threads do not inherit call stacks from their creator. This is a problem due to
our reliance on info stored on the stack. Therefore we have a limitation:

- **Limitation**: Threads need to be started using the utility class `TP` or the
  `ThreadPoolExecutor` also defined in `utils/threading.py` in order for
  instrumented methods called in a thread to be tracked. As we rely on call
  stack for call instrumentation we need to preserve the stack before a thread
  start which python does not do. 

#### Async

Similar to threads, code run as part of a `asyncio.Task` does not inherit the
stack of the creator. Our current solution instruments `asyncio.new_event_loop`
to make sure all tasks that get created in `async` track the stack of their
creator. This is done in `utils/python.py:_new_event_loop` . The function
`stack_with_tasks` is then used to integrate this information with the normal
caller stack when needed. This may cause incompatibility issues when other tools
use their own event loops or interfere with this instrumentation in other ways.
Note that some async functions that seem to not involve `Task` do use tasks,
such as `gather`.

- **Limitation**: `async.Tasks` must be created via our `task_factory` as per
  `utils/python.py:task_factory_with_stack`. This includes tasks created by
  function such as `gather`. This limitation is not expected to be a problem
  given our instrumentation except if other tools are used that modify `async`
  in some ways.

#### Limitations

- Threading and async limitations. See **Threads** and **Async** .

- If the same wrapped sub-app is called multiple times within a single call to
  the root app, the record of this execution will not be exact with regards to
  the path to the call information. All call paths will address the last subapp
  (by order in which it is instrumented). For example, in a sequential app
  containing two of the same app, call records will be addressed to the second
  of the (same) apps and contain a list describing calls of both the first and
  second.

  TODO(piotrm): This might have been fixed. Check.

- Some apps cannot be serialized/jsonized. Sequential app is an example. This is
  a limitation of langchain itself.

- Instrumentation relies on CPython specifics, making heavy use of the `inspect`
  module which is not expected to work with other Python implementations.

#### Alternatives

- langchain/llama_index callbacks. These provide information about component
  invocations but the drawbacks are need to cover disparate callback systems and
  possibly missing information not covered.

### Calls: Implementation Details

Our tracking of calls uses instrumentated versions of methods to manage the
recording of inputs/outputs. The instrumented methods must distinguish
themselves from invocations of apps that are being tracked from those not being
tracked, and of those that are tracked, where in the call stack a instrumented
method invocation is. To achieve this, we rely on inspecting the python call
stack for specific frames:
  
- Prior frame -- Each instrumented call searches for the topmost instrumented
  call (except itself) in the stack to check its immediate caller (by immediate
  we mean only among instrumented methods) which forms the basis of the stack
  information recorded alongside the inputs/outputs.

#### Drawbacks

- Python call stacks are implementation dependent and we do not expect to
  operate on anything other than CPython.

- Python creates a fresh empty stack for each thread. Because of this, we need
  special handling of each thread created to make sure it keeps a hold of the
  stack prior to thread creation. Right now we do this in our threading utility
  class TP but a more complete solution may be the instrumentation of
  threading.Thread class.

#### Alternatives

- `contextvars` -- langchain uses these to manage contexts such as those used
  for instrumenting/tracking LLM usage. These can be used to manage call stack
  information like we do. The drawback is that these are not threadsafe or at
  least need instrumenting thread creation. We have to do a similar thing by
  requiring threads created by our utility package which does stack management
  instead of contextvar management.

"""

import dataclasses
from datetime import datetime
import functools
import inspect
from inspect import BoundArguments
import logging
import os
from pprint import PrettyPrinter
import threading as th
import traceback
from typing import (
    Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple
)
import weakref

import pydantic

from trulens_eval.feedback import Feedback
from trulens_eval.feedback.provider.endpoint import Endpoint
from trulens_eval.schema import Cost
from trulens_eval.schema import Perf
from trulens_eval.schema import Record
from trulens_eval.schema import RecordAppCall
from trulens_eval.schema import RecordAppCallMethod
from trulens_eval.utils.asynchro import desync
from trulens_eval.utils.asynchro import sync
from trulens_eval.utils.containers import dict_merge_with
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import clean_attributes
from trulens_eval.utils.pyschema import Method
from trulens_eval.utils.pyschema import safe_getattr
from trulens_eval.utils.python import caller_frame
from trulens_eval.utils.python import get_first_local_in_call_stack
from trulens_eval.utils.python import is_really_coroutinefunction
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import safe_signature
from trulens_eval.utils.serial import Lens

logger = logging.getLogger(__name__)
pp = PrettyPrinter()


class WithInstrumentCallbacks:
    """
    Callbacks invoked by Instrument during instrumentation or when instrumented
    methods are called. Needs to be mixed into App.
    """

    # Called during instrumentation.
    def _on_method_instrumented(self, obj: object, func: Callable, path: Lens):
        """
        Called by instrumentation system for every function requested to be
        instrumented. Given are the object of the class in which `func` belongs
        (i.e. the "self" for that function), the `func` itsels, and the `path`
        of the owner object in the app hierarchy.
        """

        raise NotImplementedError

    # Called during invocation.
    def _get_method_path(self, obj: object, func: Callable) -> Lens:
        """
        Get the path of the instrumented function `func`, a member of the class
        of `obj` relative to this app.
        """

        raise NotImplementedError

    # WithInstrumentCallbacks requirement
    def _get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, Lens]]:
        """
        Get the methods (rather the inner functions) matching the given `func`
        and the path of each.
        """

        raise NotImplementedError

    # Called during invocation.
    def _on_new_record(self, func):
        """
        Called by instrumented methods in cases where they cannot find a record
        call list in the stack. If we are inside a context manager, return a new
        call list.
        """

        raise NotImplementedError

    # Called during invocation.
    def _on_add_record(
        self,
        record: Sequence[Record],
    ):
        """
        Called by instrumented methods if they are root calls (first instrumned
        methods in a call stack).
        """

        raise NotImplementedError


class Instrument(object):
    # TODO: might have to be made serializable soon.

    # Attribute name to be used to flag instrumented objects/methods/others.
    INSTRUMENT = "__tru_instrumented"

    APPS = "__tru_apps"

    class Default:
        # Default instrumentation configuration. Additional components are
        # included in subclasses of `Instrument`.

        # Modules (by full name prefix) to instrument.
        MODULES = {"trulens_eval."}

        # Classes to instrument.
        CLASSES = set([Feedback])

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
        return any(isinstance(obj, cls) for cls in self.include_classes)

    def to_instrument_class(self, cls: type) -> bool:  # class
        """
        Determine whether the given class should be instrumented.
        """

        # Sometimes issubclass is not supported so we return True just to be
        # sure we instrument that thing.

        try:
            return any(
                issubclass(cls, parent) for parent in self.include_classes
            )
        except Exception:
            return True

    def to_instrument_module(self, module_name: str) -> bool:
        """
        Determine whether a module with the given (full) name should be
        instrumented.
        """

        return any(
            module_name.startswith(mod2) for mod2 in self.include_modules
        )

    def __init__(
        self,
        include_modules: Iterable[str] = [],
        include_classes: Iterable[type] = [],
        include_methods: Dict[str, Callable] = {},
        app: WithInstrumentCallbacks = None
    ):

        self.include_modules = Instrument.Default.MODULES.union(
            set(include_modules)
        )

        self.include_classes = Instrument.Default.CLASSES.union(
            set(include_classes)
        )

        self.include_methods = dict_merge_with(
            dict1=Instrument.Default.METHODS,
            dict2=include_methods,
            merge=lambda f1, f2: lambda o: f1(o) or f2(o)
        )

        self.app = app

    def tracked_method_wrapper(
        self, query: Lens, func: Callable, method_name: str, cls: type,
        obj: object
    ):
        """
        Instrument a method to capture its inputs/outputs/errors.
        """

        assert not safe_hasattr(
            func, "__func__"
        ), "Function expected but method received."

        if safe_hasattr(func, Instrument.INSTRUMENT):
            logger.debug(f"\t\t\t{query}: {func} is already instrumented")

            # Notify the app instrumenting this method where it is located. Note
            # we store the method being instrumented in the attribute
            # Instrument.INSTRUMENT of the wrapped variant.
            original_func = getattr(func, Instrument.INSTRUMENT)

            self.app._on_method_instrumented(obj, original_func, path=query)

            # Add self.app, the app requesting this method to be
            # instrumented, to the list of apps expecting to be notified of
            # calls.
            existing_apps = getattr(func, Instrument.APPS)
            existing_apps.add(self.app)  # weakref set

            return func

            # TODO: How to consistently address calls to chains that appear more
            # than once in the wrapped chain or are called more than once.

        else:
            # Notify the app instrumenting this method where it is located:

            self.app._on_method_instrumented(obj, func, path=query)

        logger.debug(f"\t\t\t{query}: instrumenting {method_name}={func}")

        sig = safe_signature(func)

        def find_instrumented(f):
            # Used for finding the wrappers methods in the call stack. Note that
            # the sync version uses the async one internally and all of the
            # relevant locals are in the async version. We thus don't look for
            # sync tru_wrapper calls in the stack.
            return id(f) == id(tru_awrapper.__code__)

        @functools.wraps(func)
        async def tru_awrapper(*args, **kwargs):
            logger.debug(
                f"{query}: calling instrumented method {func} of type {type(func)}, "
                f"iscoroutinefunction={is_really_coroutinefunction(func)}, "
                f"isasyncgeneratorfunction={inspect.isasyncgenfunction(func)}"
            )

            apps = getattr(tru_awrapper, Instrument.APPS)  # DIFF

            # If not within a root method, call the wrapped function without
            # any recording.

            # Get any contexts already known from higher in the call stack.
            contexts = get_first_local_in_call_stack(
                key="contexts",
                func=find_instrumented,
                offset=1,
                skip=caller_frame()
            )
            # Note: are empty sets false?
            if contexts is None:
                contexts = set([])

            # And add any new contexts from all apps wishing to record this
            # function. This may produce some of the same contexts that were
            # already being tracked which is ok. Importantly, this might produce
            # contexts for apps that didn't instrument a method higher in the
            # call stack hence this might be the first time they are seeing an
            # instrumented method being called.
            for app in apps:
                for ctx in app._on_new_record(func):
                    contexts.add(ctx)

            if len(contexts) == 0:
                # If no app wants this call recorded, run and return without
                # instrumentation.
                logger.debug(
                    f"{query}: no record found or requested, not recording."
                )

                return await desync(func, *args, **kwargs)

            # If a wrapped method was called in this call stack, get the prior
            # calls from this variable. Otherwise create a new chain stack. As
            # another wrinke, the addresses of methods in the stack may vary
            # from app to app that are watching this method. Hence we index the
            # stacks by id of the call record list which is unique to each app.
            ctx_stacks = get_first_local_in_call_stack(
                key="stacks",
                func=find_instrumented,
                offset=1,
                skip=caller_frame()
            )
            # Note: Empty dicts are false.
            if ctx_stacks is None:
                ctx_stacks = dict()

            error = None
            rets = None

            # My own stacks to be looked up by further subcalls by the logic
            # right above. We make a copy here since we need subcalls to access
            # it but we don't want them to modify it.
            stacks = {k: v for k, v in ctx_stacks.items()}

            start_time = None
            end_time = None

            bindings = None
            cost = Cost()

            # Prepare stacks with call information of this wrapped method so
            # subsequent (inner) calls will see it. For every root_method in the
            # call stack, we make a call record to add to the existing list
            # found in the stack. Path stored in `query` of this method may
            # differ between apps that use it so we have to create a seperate
            # frame identifier for each, and therefore the stack. We also need
            # to use a different stack for the same reason. We index the stack
            # in `stacks` via id of the (unique) list `record`.

            # First prepare the stacks for each context.
            for ctx in contexts:
                # Get app that has instrumented this method.
                app = ctx.app

                # The path to this method according to the app.
                path = app._get_method_path(
                    args[0], func
                )  # hopefully args[0] is self, owner of func

                if path is None:
                    logger.warning(
                        f"App of type {type(app)} no longer knows about object 0x{id(args[0]):x} method {func}. "
                        "Something might be going wrong."
                    )
                    continue

                if ctx not in ctx_stacks:
                    # If we are the first instrumented method in the chain
                    # stack, make a new stack tuple for subsequent deeper calls
                    # (if any) to look up.
                    stack = ()
                else:
                    stack = ctx_stacks[ctx]

                frame_ident = RecordAppCallMethod(
                    path=path, method=Method.of_method(func, obj=obj, cls=cls)
                )

                stack = stack + (frame_ident,)

                stacks[ctx] = stack  # for deeper calls to get

            # Now we will call the wrapped method. We only do so once.

            # Start of run wrapped block.
            start_time = datetime.now()

            try:
                # Using sig bind here so we can produce a list of key-value
                # pairs even if positional arguments were provided.
                bindings: BoundArguments = sig.bind(*args, **kwargs)

                rets, cost = await Endpoint.atrack_all_costs_tally(
                    func, *args, **kwargs
                )

            except BaseException as e:
                error = e
                error_str = str(e)

                logger.error(f"Error calling wrapped function {func.__name__}.")
                logger.error(traceback.format_exc())

            end_time = datetime.now()

            # Done running the wrapped function. Lets collect the results.
            # Create common information across all records.

            # Don't include self in the recorded arguments.
            nonself = {
                k: jsonify(v)
                for k, v in
                (bindings.arguments.items() if bindings is not None else {})
                if k != "self"
            }

            record_app_args = dict(
                args=nonself,
                perf=Perf(start_time=start_time, end_time=end_time),
                pid=os.getpid(),
                tid=th.get_native_id(),
                rets=jsonify(rets),
                error=error_str if error is not None else None
            )
            # End of run wrapped block.

            # Now record calls to each context.
            for ctx in contexts:
                stack = stacks[ctx]

                # Note that only the stack differs between each of the records in this loop.
                record_app_args['stack'] = stack
                call = RecordAppCall(**record_app_args)
                ctx.add_call(call)

                # If stack has only 1 thing on it, we are looking at a "root
                # call". Create a record of the result and notify the app:

                if len(stack) == 1:
                    # If this is a root call, notify app to add the completed record
                    # into its containers:
                    ctx.app._on_add_record(
                        ctx=ctx,
                        func=func,
                        sig=sig,
                        bindings=bindings,
                        ret=rets,
                        error=error,
                        perf=Perf(start_time=start_time, end_time=end_time),
                        cost=cost
                    )

            if error is not None:
                raise error

            return rets

        @functools.wraps(func)
        def tru_wrapper(*args, **kwargs):
            logger.debug(
                f"{query}: calling instrumented sync method {func} of type {type(func)}, "
                f"iscoroutinefunction={is_really_coroutinefunction(func)}, "
                f"isasyncgeneratorfunction={inspect.isasyncgenfunction(func)}"
            )

            return sync(tru_awrapper, *args, **kwargs)

        # Create a new set of apps expecting to be notified about calls to the
        # instrumented method. Making this a weakref set so that if the
        # recorder/app gets garbage collected, it will be evicted from this set.
        apps = weakref.WeakSet([self.app])

        for w in [tru_wrapper, tru_awrapper]:
            # Indicate that the wrapper is an instrumented method so that we dont
            # further instrument it in another layer accidentally.
            setattr(w, Instrument.INSTRUMENT, func)
            setattr(w, Instrument.APPS, apps)

            # Set these attributes in both sync and async version because the
            # first check for something being wrapped is done on the returned
            # method and one of the two is returned below.

        # Return either the sync version or async depending on the type of func.
        if is_really_coroutinefunction(func):
            return tru_awrapper
        else:
            return tru_wrapper

    def instrument_method(self, method_name: str, obj: Any, query: Lens):
        cls = type(obj)

        logger.debug(f"{query}: instrumenting {method_name} on obj {obj}")

        for base in list(cls.__mro__):
            logger.debug(f"\t{query}: instrumenting base {base.__name__}")

            for method_name in [method_name]:

                if safe_hasattr(base, method_name):
                    original_fun = getattr(base, method_name)

                    logger.debug(
                        f"\t\t{query}: instrumenting {base.__name__}.{method_name}"
                    )
                    setattr(
                        base, method_name,
                        self.tracked_method_wrapper(
                            query=query,
                            func=original_fun,
                            method_name=method_name,
                            cls=base,
                            obj=obj
                        )
                    )

    def instrument_class(self, cls):
        """
        Instrument the given class `cls`'s __new__ method so we can be aware
        when new instances are created. This is needed for wrapped methods that
        dynamically create instances of classes we wish to instrument. As they
        will not be visible at the time we wrap the app, we need to pay
        attention to __new__ to make a note of them when they are created and
        the creator's path. This path will be used to place these new instances
        in the app json structure.
        """

        func = cls.__new__

        if safe_hasattr(func, Instrument.INSTRUMENT):
            logger.debug(
                f"Class {cls.__name__} __new__ is already instrumented."
            )
            return

        # @functools.wraps(func)
        def wrapped_new(cls, *args, **kwargs):
            logger.debug(
                f"Creating a new instance of instrumented class {cls.__name__}."
            )
            # get deepest wrapped method here
            # get its self
            # get its path
            obj = func(cls)
            # for every tracked method, and every app, do this:
            # self.app._on_method_instrumented(obj, original_func, path=query)
            return obj

        cls.__new__ = wrapped_new

    def instrument_object(
        self, obj, query: Lens, done: Optional[Set[int]] = None
    ):

        done = done or set([])

        cls = type(obj)

        mro = list(cls.__mro__)
        # Warning: cls.__mro__ sometimes returns an object that can be iterated through only once.

        logger.debug(
            f"{query}: instrumenting object at {id(obj):x} of class {cls.__name__} with mro:\n\t"
            + '\n\t'.join(map(str, mro))
        )

        if id(obj) in done:
            logger.debug(f"\t{query}: already instrumented")
            return

        done.add(id(obj))

        # NOTE: We cannot instrument chain directly and have to instead
        # instrument its class. The pydantic.BaseModel does not allow instance
        # attributes that are not fields:
        # https://github.com/pydantic/pydantic/blob/11079e7e9c458c610860a5776dc398a4764d538d/pydantic/main.py#LL370C13-L370C13
        # .

        # Recursively instrument inner components
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                if any(isinstance(attr_value, cls) for cls in self.include_classes):
                    inner_query = query[attr_name]
                    self.instrument_object(attr_value, inner_query, done)

        for base in mro:
            logger.debug(f"\t{query}: considering base {base.__name__}")

            # Some top part of mro() may need instrumentation here if some
            # subchains call superchains, and we want to capture the
            # intermediate steps. On the other hand we don't want to instrument
            # the very base classes such as object:
            if not self.to_instrument_module(base.__module__):
                logger.debug(
                    f"\tSkipping base; module {base.__module__} is not to be instrumented."
                )
                continue

            try:
                if not self.to_instrument_class(base):
                    logger.debug(
                        f"\t\tSkipping base; class {base.__name__} is not to be instrumented."
                    )
                    continue

            except Exception as e:
                # subclass check may raise exception
                logger.debug(
                    f"\t\tWarning: checking whether {base.__name__} should be instrumented resulted in an error: {e}"
                )
                # NOTE: Proceeding to instrument here as we don't want to miss
                # anything. Unsure why some llama_index subclass checks fail.

                # continue

            for method_name in self.include_methods:

                if safe_hasattr(base, method_name):
                    check_class = self.include_methods[method_name]
                    if not check_class(obj):
                        continue
                    original_fun = getattr(base, method_name)

                    # Sometimes the base class may be in some module but when a
                    # method is looked up from it, it actually comes from some
                    # other, even baser class which might come from builtins
                    # which we want to skip instrumenting.
                    if safe_hasattr(original_fun, "__self__"):
                        if not self.to_instrument_module(
                                original_fun.__self__.__class__.__module__):
                            continue
                    else:
                        # Determine module here somehow.
                        pass

                    logger.debug(f"\t\t{query}: instrumenting {method_name}")
                    setattr(
                        base, method_name,
                        self.tracked_method_wrapper(
                            query=query,
                            func=original_fun,
                            method_name=method_name,
                            cls=base,
                            obj=obj
                        )
                    )

        if self.to_instrument_object(obj) or isinstance(obj,
                                                        (dict, list, tuple)):
            vals = None
            if isinstance(obj, dict):
                attrs = obj.keys()
                vals = obj.values()

            if isinstance(obj, pydantic.BaseModel):
                # NOTE(piotrm): This will not include private fields like
                # llama_index's LLMPredictor._llm which might be useful to
                # include:
                attrs = obj.model_fields.keys()

            if isinstance(obj, pydantic.v1.BaseModel):
                attrs = obj.__fields__.keys()

            elif dataclasses.is_dataclass(type(obj)):
                attrs = (f.name for f in dataclasses.fields(obj))

            else:
                # If an object is not a recognized container type, we check that it
                # is meant to be instrumented and if so, we  walk over it manually.
                # NOTE: some llama_index objects are using dataclasses_json but most do
                # not so this section applies.
                attrs = clean_attributes(obj, include_props=True).keys()

            if vals is None:
                vals = [safe_getattr(obj, k, get_prop=True) for k in attrs]

            for k, v in zip(attrs, vals):

                if isinstance(v, (str, bool, int, float)):
                    pass

                elif self.to_instrument_module(type(v).__module__):
                    self.instrument_object(obj=v, query=query[k], done=done)

                elif isinstance(v, Sequence):
                    for i, sv in enumerate(v):
                        if self.to_instrument_class(type(sv)):
                            self.instrument_object(
                                obj=sv, query=query[k][i], done=done
                            )

                elif isinstance(v, Dict):
                    for k2, sv in v.items():
                        if self.to_instrument_class(type(sv)):
                            self.instrument_object(
                                obj=sv, query=query[k][k2], done=done
                            )

                else:
                    logger.debug(
                        f"Instrumentation of component {v} (of type {type(v)}) is not yet supported."
                    )

                # TODO: check if we want to instrument anything in langchain not
                # accessible through model_fields .

        else:
            logger.debug(
                f"{query}: Do not know how to instrument object of type {cls}."
            )


class AddInstruments():
    """
    Utilities for adding more things to default instrumentation filters.
    """

    @classmethod
    def method(self_class, cls: type, name: str) -> None:
        # Add the class with a method named `name`, its module, and the method
        # `name` to the Default instrumentation walk filters.
        Instrument.Default.MODULES.add(cls.__module__)
        Instrument.Default.CLASSES.add(cls)

        check_o = Instrument.Default.METHODS.get(name, lambda o: False)
        Instrument.Default.METHODS[
            name] = lambda o: check_o(o) or isinstance(o, cls)

    @classmethod
    def methods(self_class, cls: type, names: Iterable[str]) -> None:
        for name in names:
            self_class.method(cls, name)


class instrument(AddInstruments):
    """
    Decorator for marking methods to be instrumented in custom classes that are
    wrapped by App.
    """

    # https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class

    def __init__(self, func: Callable):
        self.func = func

    def __set_name__(self, cls: type, name: str):
        """
        For use as method decorator.
        """

        # Important: do this first:
        setattr(cls, name, self.func)

        # Note that this does not actually change the method, just adds it to
        # list of filters.
        self.method(cls, name)
