import os
from collections import defaultdict
from dataclasses import dataclass, replace, asdict
from datetime import datetime
from inspect import BoundArguments, signature, stack
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import threading as th
import multiprocessing as mp
import pydantic
from pprint import PrettyPrinter

pp = PrettyPrinter()

from langchain.chains.base import Chain

# Addresses of chains or their contents. This is used to refer chains/parameters
# even in cases where the live object is not in memory (i.e. on some remote
# app).
Address = Tuple[Union[Any, str, int], ...]

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Selection():
    model: Address = () # root chain, the argument to TruChain.__init__

    param: Address = None # model parameter

    record: Address = None 

@dataclass
class ChainCall:
    """
    Record of the execution of a single chain.
    """

    # inputs
    input: Dict[str, Any]

    # output if successful call
    output: Any = None

    # exception if not successful
    error: BaseException = None

    stack: List[str] = None

    # runtime info
    start_time: datetime = None
    end_time: datetime = None

    # process id for debugging multiprocessing
    pid: int = None

    # thread id for debuggin threading
    tid: int = None

    def __str__(self, tab=""):
        sub_tab = tab + "  "
        ret = f"{self.pid}/{self.tid} {self.start_time} - {self.end_time}\n"
        ret += f"{tab}input:\n"
        ret += f"{sub_tab}{pp.pformat(self.input)}"
        if self.output is not None:
            ret += f"\n{tab}output:\n"
            ret += f"{sub_tab}{pp.pformat(self.output)}"

        if self.stack is not None:
            ret += f"\n{tab}chain call stack:\n"
            ret += f"{sub_tab}{pp.pformat(self.stack)}"

        return ret

    def __repr__(self):
        return str(self)


@dataclass
class ObjDef:
    """
    Chain definition. May not need this in the future once chains and their
    dependencies become more consistently serializable.
    """

    # class name
    cls: str

    # module name
    module: str

    # value for basic objects that do not need serialization
    value: Any = None

    # serialization for objects that can be serialized
    dump: bytes = None

    # id for debugging scenarios where the same chain is included in a superchain more than once
    ident: int = -1

    # address of the object for referring to it even when we don't have the live object
    address: Address = None

    # fields for objects that expose fields, i.e. pydantic.BaseModel which langchain.Chain uses.
    fields: Dict[str, Any] = None

    def __str__(self, tab=""):
        sub_tab = tab + "  "
        ret = f"{self.module}.{self.cls} (at {self.address}, loaded at {self.ident})"
        if self.value is not None:
            ret += f" = {self.value}"
        if self.fields is not None:
            for f, v in self.fields.items():
                if isinstance(v, ObjDef):
                    ret += f"\n{sub_tab}{f}: {v.__str__(sub_tab)}"
                else:
                    ret += f"\n{sub_tab}{f} = {pp.pformat(v)}"

        return ret

    def __repr__(self):
        return str(self)


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
    recording: bool = False

    # Store records here.
    records: List[Dict[Address, List[ChainCall]]] = []

    # Store records as dicts here.
    # record_dicts: List[Dict[Address, List[ChainCall]]] = []

    # Store model definition here.
    model: ObjDef = None

    # Store model definition as a dictionary for remote apps.
    model_dict: Dict = None

    def __init__(self, chain: Chain):
        """
        Wrap a chain for monitoring.

        Arguments:
        - chain: Chain -- the chain to wrap.
        """

        Chain.__init__(self)

        self.chain = chain

        self._instrument(self.chain, ())
        self.model = self._current_model()
        self.model_dict = asdict(self.model)

    # Chain requirement
    @property
    def input_keys(self) -> List[str]:
        return self.chain.input_keys

    # Chain requirement
    @property
    def output_keys(self) -> List[str]:
        return self.chain.output_keys

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

        # Wrapped calls will look this up as well.
        # chain_stack = list()

        ret = None
        error = None

        try:
            ret = self.chain._call(*args, **kwargs)
        except BaseException as e:
            error = e

        self.recording = False

        assert len(record) > 0, "No information recorded in call."

        self.records.append(record)

        if error is None:
            return ret
        else:
            raise error

    def _select(self, select: Union[Selection, Sequence[Selection]]):
        ret = []

        if isinstance(select, Selection):
            select = [select]

        for record in self.records:
            row = ()

            #rdict = asdict(record)
            record = {k: [asdict(vv) for vv in v] for k, v in record.items()}
            print(record)

            for s in select:
                if s.param is not None:
                    assert s.record is None, "Selection wants a model parameter and a record at the same time. Provide these as separate Selection arguments to _select instead."

                    temp = self._get_obj_at_address(s.model + s.param, obj = self)

                elif s.record is not None:
                    if s.model in record:
                        temp = record[s.model]

                        temp = self._get_obj_at_address(s.record, obj = record)

                    else:
                        temp = None

                else:
                    raise ValueError("Selection selected neither a model parameter nor a record field.")

                row = row + (temp, )

            ret.append(row)

        return ret


    def _get_obj_at_address(self, address: Address, obj=None):
        obj = obj or self.chain

        if len(address) == 0:
            return obj

        first = address[0]
        if len(address) > 1:
            rest = address[1:]
        else:
            rest = ()

        if isinstance(first, str):
            assert hasattr(
                obj, "__fields__"
            ), f"pydantic.BaseModel expected but was {type(obj).mro()}"
            assert first in obj.__fields__, f"Object has no field '{first}', it has {list(obj.__fields__.keys())}."

            return self._get_obj_at_address(rest, getattr(obj, first))

        elif isinstance(first, int):
            assert isinstance(
                obj, Sequence), f"Sequence expected but was {type(obj)}."

            assert len(
                obj
            ) > first, f"Index {first} beyond sequence lenght {len(obj)}."

            return self._get_obj_at_address(rest, obj[first])

        else:
            raise RuntimeError(
                f"Don't know how to retrieve object at address {address} relative to {obj}."
            )

    @staticmethod
    def _address_hashable(address: Address) -> Address:
        ret = ()

        for part in address:
            if isinstance(part, (str, int)):
                ret += (part, )
            else:
                ret += (part.__class__.__name__, )

        return ret

    def _current_model(self) -> Dict:
        return self.__model(self.chain, address=())

    def __model(self, obj, address, indexed: Dict = None) -> Any:
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, int):
            return obj

        indexed = indexed or dict()

        obj_id = id(obj)

        # if obj_id in indexed:
        #     return indexed[obj_id]

        cdef = ObjDef(cls=obj.__class__.__name__,
                      module=obj.__class__.__module__,
                      address=self._address_hashable(address),
                      ident=obj_id)
        
        indexed[obj_id] = cdef

        if isinstance(obj, pydantic.BaseModel):
            subdefs = dict()

            for f in obj.__fields__:  # pdantic.BaseModel
                v = getattr(obj, f)

                if id(v) in indexed:
                    continue
                # print(f, obj_id, id(v))

                subdefs[f] = self.__model(v, address=address + (f, ), indexed=indexed)

            cdef = replace(cdef, fields=subdefs)

            indexed[obj_id] = cdef

            return cdef

        elif isinstance(obj, Sequence):
            subdefs = []

            for i, sobj in enumerate(obj):
                # print(i, obj_id, id(sobj))
                if id(sobj) in indexed:
                    continue

                subdefs.append(self.__model(sobj, address + (i, ), indexed=indexed))

            cdef = replace(cdef, value=subdefs)

            indexed[obj_id] = cdef

            return subdefs

        else: # not isinstance(obj, pydantic.BaseModel):
            cdef = replace(cdef, value=obj)
            return cdef

        

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

                key = self._address_hashable(address)
                chain_stack = self._get_local_in_call_stack(
                    key="chain_stack", func=wrapper, offset=1) or []
                chain_stack = chain_stack + [key]  # args[0] is self
                #chain_stack = []

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
                row = ChainCall(input=nonself,
                                start_time=start_time,
                                end_time=end_time,
                                pid=os.getpid(),
                                tid=th.get_native_id(),
                                stack=chain_stack)

                if error is not None:
                    row = replace(row, error=error)
                else:
                    row = replace(row, output=ret)

                record[key].append(row)

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
        # dicts but we want to capture their class information here.
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
