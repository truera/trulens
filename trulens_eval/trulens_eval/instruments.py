"""
# Instrumentation

This module contains the core of the app instrumentation scheme employed by
trulens_eval to track and record apps. These details should not be relevant for
typical use cases.
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
import threading as th
import traceback
from typing import (
    Any, Awaitable, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple,
    Type, TypeVar, Union
)
import weakref

import pydantic

from trulens_eval import trace as mod_trace
from trulens_eval.feedback import feedback as mod_feedback
from trulens_eval.feedback.provider import endpoint as mod_endpoint
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.schema import types as mod_types_schema
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
from trulens_eval.utils.serial import Lens
from trulens_eval.utils.text import retab
from trulens_eval.utils.wrap import CallableCallbacks
from trulens_eval.utils.wrap import wrap_awaitable
from trulens_eval.utils.wrap import wrap_callable

T = TypeVar("T")

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
        perf: mod_base_schema.Perf,
        cost: mod_base_schema.Cost,
        existing_record: Optional[mod_record_schema.Record] = None
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


ClassFilter = Union[Type, Tuple[Type, ...]]


def class_filter_disjunction(f1: ClassFilter, f2: ClassFilter) -> ClassFilter:
    """Create a disjunction of two class filters.
    
    Args:
        f1: The first filter.
        
        f2: The second filter.
    """

    if not isinstance(f1, Tuple):
        f1 = (f1,)

    if not isinstance(f2, Tuple):
        f2 = (f2,)

    return f1 + f2


def class_filter_matches(f: ClassFilter, obj: Union[Type, object]) -> bool:
    """Check whether given object matches a class-based filter.
    
    A class-based filter here means either a type to match against object
    (isinstance if object is not a type or issubclass if object is a type),
    or a tuple of types to match against interpreted disjunctively.

    Args:
        f: The filter to match against. 

        obj: The object to match against. If type, uses `issubclass` to
            match. If object, uses `isinstance` to match against `filters`
            of `Type` or `Tuple[Type]`.
    """

    if isinstance(f, Type):
        if isinstance(obj, Type):
            return issubclass(obj, f)

        return isinstance(obj, f)

    if isinstance(f, Tuple):
        return any(class_filter_matches(f, obj) for f in f)

    raise ValueError(f"Invalid filter {f}. Type, or a Tuple of Types expected.")


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

        CLASSES = set([mod_feedback.Feedback])
        """Classes to instrument."""

        METHODS: Dict[str, ClassFilter] = {"__call__": mod_feedback.Feedback}
        """Methods to instrument.
        
        Methods matching name have to pass the filter to be instrumented.
        """

    def print_instrumentation(self) -> None:
        """Print out description of the modules, classes, methods this class
        will instrument."""

        t = "  "

        for mod in sorted(self.include_modules):
            print(f"Module {mod}*")

            for cls in sorted(self.include_classes, key=lambda c:
                              (c.__module__, c.__qualname__)):
                if not cls.__module__.startswith(mod):
                    continue

                if isinstance(cls, Dummy):
                    print(
                        f"{t*1}Class {cls.__module__}.{cls.__qualname__}\n{t*2}WARNING: this class could not be imported. It may have been (re)moved. Error:"
                    )
                    print(retab(tab=f"{t*3}> ", s=str(cls.original_exception)))
                    continue

                print(f"{t*1}Class {cls.__module__}.{cls.__qualname__}")

                for method, class_filter in self.include_methods.items():
                    if class_filter_matches(f=class_filter,
                                            obj=cls) and safe_hasattr(cls,
                                                                      method):
                        f = getattr(cls, method)
                        print(f"{t*2}Method {method}: {inspect.signature(f)}")

            print()

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
        include_methods: Optional[Dict[str, ClassFilter]] = None,
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
            merge=class_filter_disjunction
        )

        self.app = app

    def tracked_method_wrapper(
        self, query: Lens, func: Callable, method_name: str, cls: type,
        obj: object
    ) -> Callable:
        """Wrap a method to capture its inputs/outputs/errors."""

        if self.app is None:
            raise ValueError("Instrumentation requires an app but is None.")

        if safe_hasattr(func, "__func__"):
            raise ValueError("Function expected but method received.")

        if safe_hasattr(func, Instrument.INSTRUMENT):
            logger.debug("\t\t\t%s: %s is already instrumented", query, func)

        # Notify the app instrumenting this method where it is located:
        self.app.on_method_instrumented(obj, func, path=query)

        logger.debug("\t\t\t%s: instrumenting %s=%s", query, method_name, func)

        class InstrumentationCallbacks(CallableCallbacks):

            def __init__(
                self, app: Any, query: Lens, method_name: str, cls: type,
                obj: object, sig: inspect.Signature, **kwargs: Dict[str, Any]
            ):
                super().__init__(**kwargs)

                print("init callbacks for ", query, method_name)

                self.app = app
                self.query = query
                self.method_name = method_name
                self.cls = cls
                self.obj = obj
                self.sig = sig

                tracer = mod_trace.get_tracer()
                self.span_context = tracer.method()
                self.span = self.span_context.__enter__()

                self.app_contexts = app.on_new_record(func)
                # TODO: remove

            def on_callable_end(self):
                print("exiting callbacks for ", query, method_name)

                frame_ident = mod_record_schema.RecordAppCallMethod(
                    path=self.query,
                    method=Method.of_method(
                        self.func, obj=self.obj, cls=self.cls
                    )
                )
                stack = (
                    frame_ident,
                )  # rest to be filled in later from collected spans

                nonself = {
                    k: jsonify(v) for k, v in (
                        self.bindings.arguments.items() if self.
                        bindings is not None else {}
                    ) if k != "self"
                }

                record_app_args = dict(
                    call_id=str(self.call_id),
                    args=nonself,
                    perf=mod_base_schema.Perf(
                        start_time=self.start_time, end_time=self.end_time
                    ),
                    pid=os.getpid(),
                    tid=th.get_native_id(),
                    rets=jsonify(self.ret),
                    error=str(self.error) if self.error is not None else None
                )

                #for ctx in self.app_contexts:
                record_app_args['stack'] = stack
                call = mod_record_schema.RecordAppCall(**record_app_args)
                # ctx.add_call(call)
                self.span.call = call

                # TODO: remove
                for ctx in self.app_contexts:
                    # Notify apps if this was a root call.
                    if self.span.is_root():
                        self.app.on_add_root_span(ctx=ctx, span=self.span)

                if self.error is not None:
                    self.span_context.__exit__(
                        type(self.error), self.error, self.error.__traceback__
                    )
                else:
                    self.span_context.__exit__(None, None, None)

        print("creating wrapped callable for ", query, method_name)

        return wrap_callable(
            func=func,
            callback_class=InstrumentationCallbacks,
            call_selfid=id(obj),
            app=self.app,
            query=query,
            method_name=method_name,
            cls=cls,
            obj=obj,
            sig=safe_signature(func)
        )

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

            for method_name, class_filter in self.include_methods.items():

                if safe_hasattr(base, method_name):
                    if not class_filter_matches(f=class_filter, obj=obj):
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
                        if self.to_instrument_class(type(sv)):
                            self.instrument_object(
                                obj=sv, query=subquery, done=done
                            )

                else:
                    pass

        else:
            logger.debug(
                "%s: Do not know how to instrument object of type %s.", query,
                class_name(cls)
            )

        # Check whether bound methods are instrumented properly.

    def instrument_bound_methods(self, obj: object, query: Lens):
        """Instrument functions that may be bound methods.

        Some apps include either anonymous functions or manipulates methods that
        have self bound already. Our other instrumentation cannot handle those cases.
    
        Warning:
            Experimental work in progress.
        """

        for method_name, _ in self.include_methods.items():
            if not safe_hasattr(obj, method_name):
                pass
            else:
                method = safe_getattr(obj, method_name)
                print(f"\t{query}Looking at {method}")

                if safe_hasattr(method, "__func__"):
                    func = safe_getattr(method, "__func__")
                    print(
                        f"\t\t{query}: Looking at bound method {method_name} with func {func}"
                    )

                    if safe_hasattr(func, Instrument.INSTRUMENT):
                        print(
                            f"\t\t\t{query} Bound method {func} is instrumented."
                        )

                    else:
                        print(
                            f"\t\t\t{query} Bound method {func} is not instrumented. Trying to instrument it"
                        )

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

                            setattr(obj, method_name, bound)
                        except Exception as e:
                            logger.debug(
                                f"\t\t\t{query}: cound not instrument because {e}"
                            )

                else:
                    if safe_hasattr(method, Instrument.INSTRUMENT):
                        print(
                            f"\t\t{query} Bound method {method} is instrumented."
                        )
                    else:
                        print(
                            f"\t\t{query} Bound method {method} is NOT instrumented."
                        )


class AddInstruments():
    """Utilities for adding more things to default instrumentation filters."""

    @classmethod
    def method(cls, of_cls: type, name: str) -> None:
        """Add the class with a method named `name`, its module, and the method
        `name` to the Default instrumentation walk filters."""

        Instrument.Default.MODULES.add(of_cls.__module__)
        Instrument.Default.CLASSES.add(of_cls)

        check_o: ClassFilter = Instrument.Default.METHODS.get(name, ())
        Instrument.Default.METHODS[name] = class_filter_disjunction(
            check_o, of_cls
        )

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
