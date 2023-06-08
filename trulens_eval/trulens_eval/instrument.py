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
from trulens_eval.trulens_eval.util import get_local_in_call_stack, jsonify, noserio

logger = logging.getLogger(__name__)

class Instrument(object):

    def instrument_object(self, obj, query: Query):

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
            "get_relevant_documents": lambda o: True, # VectorStoreRetriever
            "__call__": lambda o: isinstance(o, Feedback) # Feedback
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

                tv = type(v)
                mv = tv.__module__

                if isinstance(v, str):
                    pass

                elif mv.startswith("langchain.") or mv.startswith("trulens") or mv.startswith("llama_index"):
                    self.instrument_object(obj=v, query=query[k])

                elif isinstance(v, Sequence):
                    for i, sv in enumerate(v):
                        if isinstance(sv, LangChainModel.chains.base.Chain):
                            self.instrument_object(obj=sv, query=query[k][i])

                # TODO: check if we want to instrument anything not accessible through __fields__ .
        else:
            logger.debug(
                f"Do not know how to instrument object {str(obj)[:32]} of type {cls}."
            )
            
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
        self, query: Query, func: Callable, method_name: str, cls: type, obj: object
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
                # TODO: langchain specific
                return id(f) == id(TruChain.call_with_record.__code__)

            # Look up whether TruChain._call was called earlier in the stack and
            # "record" variable was defined there. Will use that for recording
            # the wrapped call.
            # TODO: langchain specific
            record = get_local_in_call_stack(
                key="record",
                func=find_call_with_record
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
                key="chain_stack",
                func=find_instrumented,
                offset=1
            ) or ()
            frame_ident = RecordChainCallMethod(
                path=query,
                method=MethodIdent.of_method(func, obj=obj)
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
                rets = rets,
                error = error_str if error is not None else None
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


class LangChainInstrument(Instrument):
    
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

 
