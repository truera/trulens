"""
# Instrumentation

This module contains the core of the app instrumentation scheme employed by
trulens_eval to track and record apps. These details should not be relevant for
typical use cases.

## Designs and Choices

### App Data

We collect app components and parameters by walking over its structure and
producing a json reprensentation with everything we deem relevant to track. The
function [jsonify][trulens_eval.utils.json.jsonify] is the root of this process.

#### class/system specific

##### pydantic (langchain)

Classes inheriting [BaseModel][pydantic.BaseModel] come with serialization
to/from json in the form of [model_dump][pydantic.BaseModel.model_dump] and
[model_validate][pydantic.BaseModel.model_validate]. We do not use the
serialization to json part of this capability as a lot of langchain components
are tripped to fail it with a "will not serialize" message. However, we use make
use of pydantic `fields` to enumerate components of an object ourselves saving
us from having to filter out irrelevant internals that are not declared as
fields.

We make use of pydantic's deserialization, however, even for our own internal
structures (see `schema.py` for example).

##### dataclasses (no present users)

The built-in dataclasses package has similar functionality to pydantic. We
use/serialize them using their field information.

##### dataclasses_json (llama_index)

Placeholder. No present special handling.

##### generic python (portions of llama_index and all else)

#### TruLens-specific Data

In addition to collecting app parameters, we also collect:

- (subset of components) App class information:

    - This allows us to deserialize some objects. Pydantic models can be
      deserialized once we know their class and fields, for example.
    - This information is also used to determine component types without having
      to deserialize them first. 
    - See [Class][trulens_eval.utils.pyschema.Class] for details.

### Functions/Methods

Methods and functions are instrumented by overwriting choice attributes in
various classes. 

#### class/system specific

##### pydantic (langchain)

Most if not all langchain components use pydantic which imposes some
restrictions but also provides some utilities. Classes inheriting
[BaseModel][pydantic.BaseModel] do not allow defining new attributes but
existing attributes including those provided by pydantic itself can be
overwritten (like dict, for example). Presently, we override methods with
instrumented versions.

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

- `wrapt` package (see https://pypi.org/project/wrapt/)

    This is only for wrapping functions or classes to resemble their original
    but does not help us with wrapping existing methods in langchain, for
    example. We might be able to use it as part of our own wrapping scheme though.

### Calls

The instrumented versions of functions/methods record the inputs/outputs and
some additional data (see
[RecordAppCallMethod][trulens_eval.schema.RecordAppCallMethod]). As more than
one instrumented call may take place as part of a app invokation, they are
collected and returned together in the `calls` field of
[Record][trulens_eval.schema.Record].

Calls can be connected to the components containing the called method via the
`path` field of [RecordAppCallMethod][trulens_eval.schema.RecordAppCallMethod].
This class also holds information about the instrumented method.

#### Call Data (Arguments/Returns)

The arguments to a call and its return are converted to json using the same
tools as App Data (see above).

#### Tricky

- The same method call with the same `path` may be recorded multiple times in a
  `Record` if the method makes use of multiple of its versions in the class
  hierarchy (i.e. an extended class calls its parents for part of its task). In
  these circumstances, the `method` field of
  [RecordAppCallMethod][trulens_eval.schema.RecordAppCallMethod] will
  distinguish the different versions of the method.

- Thread-safety -- it is tricky to use global data to keep track of instrumented
  method calls in presence of multiple threads. For this reason we do not use
  global data and instead hide instrumenting data in the call stack frames of
  the instrumentation methods. See
  [get_all_local_in_call_stack][trulens_eval.utils.python.get_all_local_in_call_stack].

- Generators and Awaitables -- If an instrumented call produces a generator or
  awaitable, we cannot produce the full record right away. We instead create a
  record with placeholder values for the yet-to-be produce pieces. We then
  instrument (i.e. replace them in the returned data) those pieces with (TODO
  generators) or awaitables that will update the record when they get eventually
  awaited (or generated).

#### Threads

Threads do not inherit call stacks from their creator. This is a problem due to
our reliance on info stored on the stack. Therefore we have a limitation:

- **Limitation**: Threads need to be started using the utility class
  [TP][trulens_eval.utils.threading.TP] or
  [ThreadPoolExecutor][trulens_eval.utils.threading.ThreadPoolExecutor] also
  defined in `utils/threading.py` in order for instrumented methods called in a
  thread to be tracked. As we rely on call stack for call instrumentation we
  need to preserve the stack before a thread start which python does not do. 

#### Async

Similar to threads, code run as part of a [asyncio.Task][] does not inherit
the stack of the creator. Our current solution instruments
[asyncio.new_event_loop][] to make sure all tasks that get created
in `async` track the stack of their creator. This is done in
[tru_new_event_loop][trulens_eval.utils.python.tru_new_event_loop] . The
function [stack_with_tasks][trulens_eval.utils.python.stack_with_tasks] is then
used to integrate this information with the normal caller stack when needed.
This may cause incompatibility issues when other tools use their own event loops
or interfere with this instrumentation in other ways. Note that some async
functions that seem to not involve [Task][asyncio.Task] do use tasks, such as
[gather][asyncio.gather].

- **Limitation**: [Task][asyncio.Task]s must be created via our `task_factory`
  as per
  [task_factory_with_stack][trulens_eval.utils.python.task_factory_with_stack].
  This includes tasks created by function such as [asyncio.gather][]. This
  limitation is not expected to be a problem given our instrumentation except if
  other tools are used that modify `async` in some ways.

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

- Instrumentation relies on CPython specifics, making heavy use of the
  [inspect][] module which is not expected to work with other Python
  implementations.

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
  [threading.Thread][] class.

#### Alternatives

- [contextvars][] -- langchain uses these to manage contexts such as those used
  for instrumenting/tracking LLM usage. These can be used to manage call stack
  information like we do. The drawback is that these are not threadsafe or at
  least need instrumenting thread creation. We have to do a similar thing by
  requiring threads created by our utility package which does stack management
  instead of contextvar management.

    NOTE(piotrm): it seems to be standard thing to do to copy the contextvars into
    new threads so it might be a better idea to use contextvars instead of stack
    inspection.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import functools
import inspect
from inspect import BoundArguments
from inspect import Signature
import logging
import os
from pprint import pformat
import threading as th
import traceback
from typing import (Any, Awaitable, Callable, Dict, Iterable, Optional,
                    Sequence, Set, Tuple)
import weakref

import pydantic

from trulens_eval.feedback import Feedback
from trulens_eval.feedback.provider.endpoint import Endpoint
from trulens_eval.schema import Cost
from trulens_eval.schema import Perf
from trulens_eval.schema import Record
from trulens_eval.schema import RecordAppCall
from trulens_eval.schema import RecordAppCallMethod
from trulens_eval.utils import python
from trulens_eval.utils.containers import dict_merge_with
from trulens_eval.utils.imports import Dummy
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import clean_attributes
from trulens_eval.utils.pyschema import Method
from trulens_eval.utils.pyschema import safe_getattr
from trulens_eval.utils.python import callable_name
from trulens_eval.utils.python import caller_frame
from trulens_eval.utils.python import class_name
from trulens_eval.utils.python import get_first_local_in_call_stack
from trulens_eval.utils.python import id_str
from trulens_eval.utils.python import is_really_coroutinefunction
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import safe_signature
from trulens_eval.utils.python import wrap_awaitable
from trulens_eval.utils.serial import Lens

logger = logging.getLogger(__name__)


class WithInstrumentCallbacks:
    """Abstract definition of callbacks invoked by Instrument during
    instrumentation or when instrumented methods are called.
    
    Needs to be mixed into [App][trulens_eval.app.App].
    """

    # Called during instrumentation.
    def on_method_instrumented(self, obj: object, func: Callable, path: Lens):
        """Callback to be called by instrumentation system for every function
        requested to be instrumented.
        
        Given are the object of the class in which `func` belongs
        (i.e. the "self" for that function), the `func` itsels, and the `path`
        of the owner object in the app hierarchy.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).

            path: The path of the owner object in the app hierarchy.
        """

        raise NotImplementedError

    # Called during invocation.
    def get_method_path(self, obj: object, func: Callable) -> Lens:
        """
        Get the path of the instrumented function `func`, a member of the class
        of `obj` relative to this app.

        Args:
            obj: The object of the class in which `func` belongs (i.e. the
                "self" for that method).

            func: The function that was instrumented. Expects the unbound
                version (self not yet bound).
        """

        raise NotImplementedError

    # WithInstrumentCallbacks requirement
    def get_methods_for_func(
        self, func: Callable
    ) -> Iterable[Tuple[int, Callable, Lens]]:
        """
        Get the methods (rather the inner functions) matching the given `func`
        and the path of each.

        Args:
            func: The function to match.
        """

        raise NotImplementedError

    # Called during invocation.
    def on_new_record(self, func: Callable):
        """
        Called by instrumented methods in cases where they cannot find a record
        call list in the stack. If we are inside a context manager, return a new
        call list.
        """

        raise NotImplementedError

    # Called during invocation.
    def on_add_record(
        self,
        ctx: 'RecordingContext',
        func: Callable,
        sig: Signature,
        bindings: BoundArguments,
        ret: Any,
        error: Any,
        perf: Perf,
        cost: Cost,
        existing_record: Optional[Record] = None
    ):
        """
        Called by instrumented methods if they are root calls (first instrumned
        methods in a call stack).

        Args:
            ctx: The context of the recording.

            func: The function that was called.

            sig: The signature of the function.

            bindings: The bound arguments of the function.

            ret: The return value of the function.

            error: The error raised by the function if any.

            perf: The performance of the function.

            cost: The cost of the function.

            existing_record: If the record has already been produced (i.e.
                because it was an awaitable), it can be passed here to avoid
                re-creating it.
        """

        raise NotImplementedError


class Instrument(object):
    """Instrumentation tools."""

    INSTRUMENT = "__tru_instrumented"
    """Attribute name to be used to flag instrumented objects/methods/others."""

    APPS = "__tru_apps"
    """Attribute name for storing apps that expect to be notified of calls."""

    class Default:
        """Default instrumentation configuration.
        
        Additional components are included in subclasses of
        [Instrument][trulens_eval.instruments.Instrument]."""

        MODULES = {"trulens_eval."}
        """Modules (by full name prefix) to instrument."""

        CLASSES = set([Feedback])
        """Classes to instrument."""

        METHODS = {"__call__": lambda o: isinstance(o, Feedback)}
        """Methods to instrument.
        
        Methods matching name have to pass the filter to be instrumented.
        """


    def print_instrumentation(self) -> None:
        """Print out description of the modules, classes, methods this class
        will instrument."""

        print("Modules (with prefix of):")
        for mod in self.include_modules:
            print(f"\t{mod}")

        print("Classes (or subclasses of):")
        for cls in self.include_classes:
            print(f"\t{cls}")

            if isinstance(cls, Dummy):
                print("\tWARNING: this class could not be imported. It may have been removed from its library.")
                continue

            for method, filter_class in self.include_methods.items():
                if filter_class(cls) and safe_hasattr(cls, method):
                    f = getattr(cls, method)
                    print(f"\t\t{method}: {inspect.signature(f)}")

    def to_instrument_object(self, obj: object) -> bool:
        """Determine whether the given object should be instrumented."""

        # NOTE: some classes do not support issubclass but do support
        # isinstance. It is thus preferable to do isinstance checks when we can
        # avoid issublcass checks.
        return any(isinstance(obj, cls) for cls in self.include_classes)

    def to_instrument_class(self, cls: type) -> bool:  # class
        """Determine whether the given class should be instrumented."""

        # Sometimes issubclass is not supported so we return True just to be
        # sure we instrument that thing.

        try:
            return any(
                issubclass(cls, parent) for parent in self.include_classes
            )
        except Exception:
            return True

    def to_instrument_module(self, module_name: str) -> bool:
        """Determine whether a module with the given (full) name should be instrumented."""

        return any(
            module_name.startswith(mod2) for mod2 in self.include_modules
        )

    def __init__(
        self,
        include_modules: Optional[Iterable[str]] = None,
        include_classes: Optional[Iterable[type]] = None,
        include_methods: Optional[Dict[str, Callable]] = None,
        app: Optional[WithInstrumentCallbacks] = None
    ):
        if include_modules is None:
            include_modules = []
        if include_classes is None:
            include_classes = []
        if include_methods is None:
            include_methods = {}

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
        """Wrap a method to capture its inputs/outputs/errors."""

        if self.app is None:
            raise ValueError("Instrumentation requires an app but is None.")

        if safe_hasattr(func, "__func__"):
            raise ValueError("Function expected but method received.")

        if safe_hasattr(func, Instrument.INSTRUMENT):
            logger.debug("\t\t\t%s: %s is already instrumented", query, func)

            # Notify the app instrumenting this method where it is located. Note
            # we store the method being instrumented in the attribute
            # Instrument.INSTRUMENT of the wrapped variant.
            original_func = getattr(func, Instrument.INSTRUMENT)

            self.app.on_method_instrumented(obj, original_func, path=query)

            # Add self.app, the app requesting this method to be
            # instrumented, to the list of apps expecting to be notified of
            # calls.
            existing_apps = getattr(func, Instrument.APPS)
            existing_apps.add(self.app)  # weakref set

            return func

        # Notify the app instrumenting this method where it is located:
        self.app.on_method_instrumented(obj, func, path=query)

        logger.debug("\t\t\t%s: instrumenting %s=%s", query, method_name, func)

        sig = safe_signature(func)

        def find_instrumented(f):
            # Used for finding the wrappers methods in the call stack. Note that
            # the sync version uses the async one internally and all of the
            # relevant locals are in the async version. We thus don't look for
            # sync tru_wrapper calls in the stack.
            return id(f) == id(tru_wrapper.__code__)

        @functools.wraps(func)
        def tru_wrapper(*args, **kwargs):
            logger.debug(
                "%s: calling instrumented sync method %s of type %s, "
                "iscoroutinefunction=%s, "
                "isasyncgeneratorfunction=%s", query, func, type(func),
                is_really_coroutinefunction(func),
                inspect.isasyncgenfunction(func)
            )

            apps = getattr(tru_wrapper, Instrument.APPS)

            # If not within a root method, call the wrapped function without
            # any recording.

            # Get any contexts already known from higher in the call stack.
            contexts = get_first_local_in_call_stack(
                key="contexts",
                func=find_instrumented,
                offset=1,
                skip=python.caller_frame()
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
                for ctx in app.on_new_record(func):
                    contexts.add(ctx)

            if len(contexts) == 0:
                # If no app wants this call recorded, run and return without
                # instrumentation.
                logger.debug(
                    "%s: no record found or requested, not recording.", query
                )

                return func(*args, **kwargs)

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
                ctx_stacks = {}

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
                path = app.get_method_path(
                    args[0], func
                )  # hopefully args[0] is self, owner of func

                if path is None:
                    logger.warning(
                        "App of type %s no longer knows about object %s method %s. "
                        "Something might be going wrong.",
                        class_name(type(app)), id_str(args[0]),
                        callable_name(func)
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

            error_str = None

            try:
                # Using sig bind here so we can produce a list of key-value
                # pairs even if positional arguments were provided.
                bindings: BoundArguments = sig.bind(*args, **kwargs)

                rets, cost = Endpoint.track_all_costs_tally(
                    func, *args, **kwargs
                )

            except BaseException as e:
                error = e
                error_str = str(e)

                logger.error(
                    "Error calling wrapped function %s.", callable_name(func)
                )
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

            records = {}

            def handle_done(rets):
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
                        records[ctx] = ctx.app.on_add_record(
                            ctx=ctx,
                            func=func,
                            sig=sig,
                            bindings=bindings,
                            ret=rets,
                            error=error,
                            perf=Perf(start_time=start_time, end_time=end_time),
                            cost=cost,
                            existing_record=records.get(ctx)
                        )

                if error is not None:
                    raise error

                return records

            if isinstance(rets, Awaitable):
                # If method produced an awaitable, add a placeholder record
                # stating that results are not ready and return an awaitable
                # that will capture the results when they are ready.

                # Placeholder:
                records: Dict = handle_done(
                    rets=(
                        f"""
NOTE from trulens_eval:
This app produced an asynchronous response of type `{class_name(type(rets))}`. This record will be updated once
the response is available. If this message persists, check that you are
using the correct version of the app method and `await` any asynchronous
results. Additional information about this call: 
    
```
{pformat(locals())}
```
    """
                    ),
                )

                # TODO(piotrm): need to track costs of awaiting the ret in the
                # below.

                return wrap_awaitable(rets, on_done=handle_done)

            handle_done(rets=rets)
            return rets

        # Create a new set of apps expecting to be notified about calls to the
        # instrumented method. Making this a weakref set so that if the
        # recorder/app gets garbage collected, it will be evicted from this set.
        apps = weakref.WeakSet([self.app])

        # Indicate that the wrapper is an instrumented method so that we dont
        # further instrument it in another layer accidentally.
        setattr(tru_wrapper, Instrument.INSTRUMENT, func)
        setattr(tru_wrapper, Instrument.APPS, apps)

        return tru_wrapper

    def instrument_method(self, method_name: str, obj: Any, query: Lens):
        """Instrument a method."""

        cls = type(obj)

        logger.debug("%s: instrumenting %s on obj %s", query, method_name, obj)

        for base in list(cls.__mro__):
            logger.debug("\t%s: instrumenting base %s", query, class_name(base))

            if safe_hasattr(base, method_name):
                original_fun = getattr(base, method_name)

                logger.debug(
                    "\t\t%s: instrumenting %s.%s", query, class_name(base),
                    method_name
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
        """Instrument the given class `cls`'s __new__ method.
         
        This is done so we can be aware when new instances are created and is
        needed for wrapped methods that dynamically create instances of classes
        we wish to instrument. As they will not be visible at the time we wrap
        the app, we need to pay attention to __new__ to make a note of them when
        they are created and the creator's path. This path will be used to place
        these new instances in the app json structure.
        """

        func = cls.__new__

        if safe_hasattr(func, Instrument.INSTRUMENT):
            logger.debug(
                "Class %s __new__ is already instrumented.", class_name(cls)
            )
            return

        # @functools.wraps(func)
        def wrapped_new(cls, *args, **kwargs):
            logger.debug(
                "Creating a new instance of instrumented class %s.",
                class_name(cls)
            )
            # get deepest wrapped method here
            # get its self
            # get its path
            obj = func(cls)
            # for every tracked method, and every app, do this:
            # self.app.on_method_instrumented(obj, original_func, path=query)
            return obj

        cls.__new__ = wrapped_new

    def instrument_object(
        self, obj, query: Lens, done: Optional[Set[int]] = None
    ):
        """Instrument the given object `obj` and its components."""

        done = done or set([])

        cls = type(obj)

        mro = list(cls.__mro__)
        # Warning: cls.__mro__ sometimes returns an object that can be iterated through only once.

        logger.debug(
            "%s: instrumenting object at %s of class %s", query, id_str(obj),
            class_name(cls)
        )

        if id(obj) in done:
            logger.debug("\t%s: already instrumented", query)
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
                if any(isinstance(attr_value, cls)
                       for cls in self.include_classes):
                    inner_query = query[attr_name]
                    self.instrument_object(attr_value, inner_query, done)

        for base in mro:
            # Some top part of mro() may need instrumentation here if some
            # subchains call superchains, and we want to capture the
            # intermediate steps. On the other hand we don't want to instrument
            # the very base classes such as object:
            if not self.to_instrument_module(base.__module__):
                continue

            try:
                if not self.to_instrument_class(base):
                    continue

            except Exception as e:
                # subclass check may raise exception
                logger.debug(
                    "\t\tWarning: checking whether %s should be instrumented resulted in an error: %s",
                    python.module_name(base), e
                )
                # NOTE: Proceeding to instrument here as we don't want to miss
                # anything. Unsure why some llama_index subclass checks fail.

                # continue

            for method_name, check_class in self.include_methods.items():

                if safe_hasattr(base, method_name):
                    if not check_class(obj):
                        continue

                    original_fun = getattr(base, method_name)

                    # If an instrument class uses a decorator to wrap one of
                    # their methods, the wrapper will capture an uninstrumented
                    # version of the inner method which we may fail to
                    # instrument.
                    if hasattr(original_fun, "__wrapped__"):
                        original_fun = original_fun.__wrapped__

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

                    logger.debug("\t\t%s: instrumenting %s", query, method_name)

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
                        subquery = query[k][k2]
                        # WORK IN PROGRESS: BUG: some methods in rails are bound with a class that we cannot instrument
                        """
                        if isinstance(sv, Callable):
                            if safe_hasattr(sv, "__self__"):
                                # Is a method with bound self.
                                sv_self = getattr(sv, "__self__")
                                
                                if not self.to_instrument_class(type(sv_self)):
                                    # print(f"{subquery}: Don't need to instrument class {type(sv_self)}")
                                    continue

                                if not safe_hasattr(sv, self.INSTRUMENT):
                                    print(f"{subquery}: Trying to instrument bound methods in {sv_self}")

                                if safe_hasattr(sv, "__func__"):
                                    func = getattr(sv, "__func__")
                                    if not safe_hasattr(func, self.INSTRUMENT):
                                        print(f"{subquery}: Bound method {sv}, unbound {func} is not instrumented. Trying to instrument.")

                                        subobj = sv.__self__

                                        try:
                                            unbound = self.tracked_method_wrapper(
                                                query=query,
                                                func=func,
                                                method_name=func.__name__,
                                                cls=type(subobj),
                                                obj=subobj
                                            )
                                            if inspect.iscoroutinefunction(func):
                                                @functools.wraps(unbound)
                                                async def bound(*args, **kwargs):
                                                    return await unbound(subobj, *args, **kwargs)
                                            else:
                                                def bound(*args, **kwargs):
                                                    return unbound(subobj, *args, **kwargs)
                                                
                                            v[k2] = bound
                                            
                                            #setattr(
                                            #    sv.__func__, "__code__", unbound.__code__
                                            #)
                                        except Exception as e:
                                            print(f"\t\t\t{subquery}: cound not instrument because {e}")
                                                    #self.instrument_bound_methods(sv_self, query=subquery)
                                        
                        """
                        if self.to_instrument_class(type(sv)):
                            self.instrument_object(
                                obj=sv, query=subquery, done=done
                            )

                else:
                    pass

                # TODO: check if we want to instrument anything in langchain not
                # accessible through model_fields .

        else:
            logger.debug(
                "%s: Do not know how to instrument object of type %s.", query,
                class_name(cls)
            )

        # Check whether bound methods are instrumented properly.
        

    def instrument_bound_methods(self, obj: object, query: Lens):
        # TODO: Work in progress. Bugfixing rails instrumentation missing some important methods.

        for method_name, _ in self.include_methods.items():
            if not (safe_hasattr(obj, method_name) and self.include_methods[method_name](obj)):
                pass
            else:
                method = safe_getattr(obj, method_name)
                print(f"\t{query}Looking at {method}")

                if safe_hasattr(method, "__func__"):
                    func = safe_getattr(method, "__func__")
                    print(f"\t\t{query}: Looking at bound method {method_name} with func {func}")

                    if safe_hasattr(func, Instrument.INSTRUMENT):
                        print(f"\t\t\t{query} Bound method {func} is instrumented.")

                    else:
                        print(f"\t\t\t{query} Bound method {func} is not instrumented. Trying to instrument it")
                    
                        try:
                            unbound = self.tracked_method_wrapper(
                                query=query,
                                func=func,
                                method_name=method_name,
                                cls=type(obj),
                                obj=obj
                            )
                            if inspect.iscoroutinefunction(func):
                                async def bound(*args, **kwargs):
                                    return await unbound(obj, *args, **kwargs)
                            else:
                                def bound(*args, **kwargs):
                                    return unbound(obj, *args, **kwargs)
                                
                            setattr(
                                obj, method_name, bound
                            )
                        except Exception as e:
                            logger.debug(f"\t\t\t{query}: cound not instrument because {e}")
                        
                else:
                    if safe_hasattr(method, Instrument.INSTRUMENT):
                        print(f"\t\t{query} Bound method {method} is instrumented.")
                    else:
                        print(f"\t\t{query} Bound method {method} is NOT instrumented.")


class AddInstruments():
    """Utilities for adding more things to default instrumentation filters."""

    @classmethod
    def method(cls, of_cls: type, name: str) -> None:
        """Add the class with a method named `name`, its module, and the method
        `name` to the Default instrumentation walk filters."""

        Instrument.Default.MODULES.add(of_cls.__module__)
        Instrument.Default.CLASSES.add(of_cls)

        check_o = Instrument.Default.METHODS.get(name, lambda o: False)
        Instrument.Default.METHODS[
            name] = lambda o: check_o(o) or isinstance(o, of_cls)

    @classmethod
    def methods(cls, of_cls: type, names: Iterable[str]) -> None:
        """Add the class with methods named `names`, its module, and the named
        methods to the Default instrumentation walk filters."""

        for name in names:
            cls.method(of_cls, name)


class instrument(AddInstruments):
    """Decorator for marking methods to be instrumented in custom classes that are wrapped by App."""

    # NOTE(piotrm): Approach taken from:
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
