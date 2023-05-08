import json
import os
import threading as th
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from inspect import BoundArguments, signature, stack
from pprint import PrettyPrinter
from types import NoneType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tinydb import TinyDB, Query
from tinydb.table import Table


from pydantic import Field, BaseModel

pp = PrettyPrinter()

from langchain.chains.base import Chain

# Addresses of chains or their contents. This is used to refer chains/parameters
# even in cases where the live object is not in memory (i.e. on some remote
# app).
Address = Tuple[Union[Any, str, int], ...]

# from dataclasses import dataclass
from typing import Tuple

# Records of a chain run are dictionaries with these keys:
#
# - 'input': Dict[str, Any] -- chain input.
# - 'output': Any -- output for calls that succeed.
# - 'error': str -- exception text if not successful.
# - 'start_time': datetime
# - 'end_time': datetime -- runtime info.
# - 'pid': int -- process id for debugging multiprocessing.
# - 'tid': int -- thread id for debuggin threading.

import pandas as pd

Record = Query()

def select(table: Table, fields: List[Record], where: Optional[Record] = None) -> pd.DataFrame:
    rows = []

    if where is not None:
        table_rows = table.search(where)
    else:
        table_rows = table.all()

    for row in table_rows:
        vals = [project(address=f, obj=row) for f in fields]
        rows.append(vals)

    return pd.DataFrame(rows, columns=fields)

def project(query: Record, obj: Any):
    return _project(query._path, obj)

def _project(path: List, obj: Any):
    if len(path) == 0:
        return obj
    
    first = [0]
    if len(path) > 1:
        rest = path[1:]
    else:
        rest = ()

    if isinstance(first, str):
        if not isinstance(obj, Dict) or first not in obj:
            return None    
        
        return _project(rest, obj[first])
    
    elif isinstance(first, int):
        if not isinstance(obj, Sequence) or first >= len(obj):
            return None
        
        return _project(rest, obj[first])
    else:
        raise RuntimeError(f"Don't know how to locate element with key of type {first}")

class TruChain(Chain):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    
    TODO: NOT thread-safe. Do not use simulatenously from multiple threads for
    now.
    
    Example usage:
    
    ```python
        tru_chain = TruChain(chain=llm_chain)

        # Get a dictionary containing the parameters defining llm_chain such as
        # prompts. 

        tru_chain.model 

        ... calls to tru_chain as if it were llm_chain ...

        # Get a list of the calls involved in the chain capturing their inputs
        # and outputs: 
        
        tru_chain.records 
    ```
    """

    # The wrapped/instrumented chain.
    chain: Chain = None

    # Flag of whether the chain is currently recording records. This is set
    # automatically but is imperfect in threaded situations. The second check
    # for recording is based on the call stack, see _call.
    recording: Optional[bool] = Field(exclude=True)

    # Store records here. "exclude=True" means that these fields will not be
    # included in the json/dict dump.
    records: Optional[List[Dict[Address, List[Dict]]]] = Field(exclude=True)

    # TinyDB json database to write records to. Need to call _flush_records for
    # this though.
    db: Optional[tinydb.TinyDB] = Field(exclude=True)

    def __init__(self, chain: Chain, db: Optional[tinydb.TinyDB] = None):
        """
        Wrap a chain for monitoring.

        Arguments:
        - chain: Chain -- the chain to wrap.
        """

        Chain.__init__(self)

        self.chain = chain

        self._instrument(self.chain, ("chain",))
        self.recording = False
        self.records = []

        self.db = db

    @property
    def _model(self):
        return self.dict()

    def _flush_records(self, db: Optional[tinydb.TinyDB] = None):
        # TODO: locks
        
        # NOTE: TinyDB is annoyingly false.
        db = db if db is not None else self.db

        table = db.table("records")
        to_flush = self.records
        self.records = []
        for record in to_flush:
            table.insert(record)

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

    # Chain requirement
    def _call(self, *args, **kwargs) -> Any:
        """
        Wrapped call to self.chain._call with instrumentation.
        """

        # Mark us as recording calls. Should be sufficient for non-threaded cases.
        self.recording = True

        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record = defaultdict(list)

        ret = None
        error = None

        try:
            ret = self.chain._call(*args, **kwargs)
        except BaseException as e:
            error = str(e)
            print(f"WARNING: {e}")

        self.recording = False

        assert len(record) > 0, "No information recorded in call."

        model = self.dict()
        for addr, calls in record.items():
            obj = project(addr, model)
            obj.update(dict(_call=calls))

        self.records.append(model)

        if error is None:
            return ret
        else:
            raise error

    def _get_local_in_call_stack(self,
                                 key: str,
                                 func: Callable,
                                 offset: int = 1) -> Optional[Any]:
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
                    raise RuntimeError(
                        f"No local named {key} in {func} found.")

        return None
    
    def _instrument_chain_type(self, chain, prop):
        """
        Instrument the Chain class's method _chain_type which is presently used
        to control model saving. Override the exception behaviour. Note that
        _chain_type is defined as a property in langchain.
        """

        # Properties doesn't let us new define new attributes like "_instrument"
        # so we put it on fget instead.
        if hasattr(prop.fget, "_instrumented"):
            prop = prop.fget._instrumented

        def safe_chain_type(s) -> Union[str, Dict]:
            # self should be chain
            try:
                ret = prop.fget(s)
                return ret

            except NotImplementedError as e:

                ret = dict(error=e)
                ret['class'] = chain.__class__.__name__
                ret['module'] = chain.__class__.__module__

                return ret

        safe_chain_type._instrumented = prop
        new_prop = property(fget=safe_chain_type)

        return new_prop

    def _instrument_method(self, address, func):
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
            if not self.recording:
                return func(*args, **kwargs)

            # Look up whether TruChain._call was called earlier in the stack and
            # "record" variable was defined there. Will use that for recording
            # the wrapped call.
            record = self._get_local_in_call_stack(key="record",
                                                   func=TruChain._call)

            if record is None:
                return func(*args, **kwargs)

            else:
                # Otherwise keep track of inputs and outputs (or exception).

                error = None
                ret = None

                start_time = datetime.now()

                # key = self._address_hashable(address)
                chain_stack = self._get_local_in_call_stack(
                    key="chain_stack", func=wrapper, offset=1) or []
                chain_stack = chain_stack + [address]  # args[0] is self

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
                    k: v
                    for k, v in bindings.arguments.items() if k != "self"
                }
                row_args = dict(input=nonself,
                                start_time=str(start_time),
                                end_time=str(end_time),
                                pid=os.getpid(),
                                tid=th.get_native_id(),
                                chain_stack=chain_stack
                                )

                if error is not None:
                    row_args['error'] = error
                else:
                    row_args['output'] = ret

                record[address].append(row_args)

                if error is not None:
                    raise error

                return ret

        wrapper._instrumented = func

        # Put the address of the instrumented chain in the wrapper so that we
        # don't pollute its list of fields. Note that this address may be
        # deceptive if the same subchain appears multiple times in the wrapped
        # chain.
        wrapper._address = address

        return wrapper

    def _instrument(self, chain, address=()):
        if self.verbose:
            print(f"instrumenting {address} {chain.__class__.__name__}")

        c = chain.__class__

        # NOTE: We cannot instrument chain directly and have to instead
        # instrument its class. The pydantic BaseModel does not allow instance
        # attributes that are not fields:
        # https://github.com/pydantic/pydantic/blob/11079e7e9c458c610860a5776dc398a4764d538d/pydantic/main.py#LL370C13-L370C13
        # .
        for cp in [c]:
            # All of mro() may need instrumentation here if some subchains call
            # superchains, and we want to capture the intermediate steps.

            if "langchain" not in str(cp):
                continue

            if hasattr(cp, "_call"):
                original_fun = getattr(cp, "_call")

                if self.verbose:
                    print(f"instrumented {cp}._call")

                setattr(cp, "_call",
                        self._instrument_method(address, original_fun))

            if hasattr(cp, "_chain_type"):
                prop = getattr(cp, "_chain_type")
                setattr(cp, "_chain_type",
                        self._instrument_chain_type(chain=chain, prop=prop))

        # Not using chain.dict() here as that recursively converts subchains to
        # dicts but we want to traverse the instantiations here.
        for k in chain.__fields__:
            # NOTE(piotrm): may be better to use inspect.getmembers_static .
            v = getattr(chain, k)

            if isinstance(v, str):
                pass

            elif isinstance(v, Chain):
                self._instrument(v, address=address + (k, ))

            elif isinstance(v, Sequence):
                for i, sv in enumerate(v):
                    if isinstance(sv, Chain):
                        self._instrument(sv, address=address + (
                            k,
                            i,
                        ))

            # TODO: check if we want to instrument anything not accessible through __fields__ .
