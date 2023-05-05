from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from inspect import BoundArguments, signature
from typing import Any, Dict, List, Sequence, Tuple, Union

from langchain.chains.base import Chain

@dataclass
class ChainCall:
    input: Dict
    output: Any = None
    error: BaseException = None

    start_time: datetime = None
    end_time: datetime = None

class TruChain(Chain):
    """
    Wrap a langchain Chain to capture its configuration and evaluation steps. 
    
    TODO: NOT thread-safe. Do not use simulatenously from multiple threads for now.
    
    Example usage:
    ```python
        tru_chain = TruChain(chain=llm_chain)

        # Get a dictionary containing the parameters defining llm_chain such as prompts.
        tru_chain.model 

        ... calls to tru_chain as if it were llm_chain ...

        # Get a list of the calls involved in the chain capturing their inputs and outputs:
        tru_chain.records 
    ```
    """

    chain: Chain = None

    record: Dict = None
    records: List[Dict] = []
    model: Dict = None
    models: List[Dict] = []

    def __init__(self, chain: Chain):
        """
        Wrap a chain for monitoring.

        Arguments:
        - chain: Chain -- the chain to wrap.
        """

        Chain.__init__(self)

        self.chain = chain

        self._instrument(self.chain, (self.chain,))
        self.model = self._current_model()

    # Chain requirement
    @property
    def input_keys(self) -> List[str]:
        return self.chain.input_keys

    # Chain requirement
    @property
    def output_keys(self) -> List[str]:
        return self.chain.output_keys

    # Chain requirement
    def _call(self, *args, **kwargs):
        # NOT thread safe
        assert self.record is None

        self.record = defaultdict(list)
        ret = self.chain._call(*args, **kwargs)
        self.records.append(self.record)
        self.record = None

        return ret

    @staticmethod
    def _address_hashable(address) -> Tuple[str,...]:
        ret = ()

        for part in address:
            if isinstance(part, (str, int)):
                ret += (part,)
            else:
                ret += (part.__class__.__name__,)

        return ret

    def _current_model(self) -> Dict:
        return self.__model(self.chain, address=(self.chain,), conf=dict())

    def __model(self, obj, address, conf=None) -> Dict:
        conf = conf or dict()

        conf['class'] = obj.__class__.__name__
        conf['module'] = obj.__class__.__module__

        if hasattr(obj, "dict"): # pydantic.BaseModel subclasses like Chain

            conf['fields'] = obj.dict() # Recursively produces dicts here.

        else:
            # For non pydantic objects, traverse their structure and capture any
            # internal chains or simple data. Unclear whether this ever happens.

            rest = dict()
            conf['value'] = rest

            for k in dir(obj):
                if k.startswith("_"):
                    continue

                key = address + (k, )

                v = getattr(obj, k)

                if isinstance(v, str):
                    rest[key] = v

                elif isinstance(v, Chain):
                    rest[key] = dict()
                    self._model(v, address=address + (k, ), conf=rest[key])

                elif isinstance(v, Sequence):
                    rest[key] = list()

                    for i, sv in enumerate(v):
                        rest[key].append(dict())

                        if isinstance(sv, Chain):
                            self._model(sv, address=address + (k, i, ), conf=rest[key][-1])

                else:
                    print(f"WARNING: key {k} of {obj} address {address} not captured in model definition.")

        return conf

    def _instrument_chain_type(self, chain, prop):
        """
        Instrument the Chain class's method _chain_type which is presently used
        to control model saving. Override the exception behaviour. Note that
        _chain_type is defined as a property in langchain.
        """

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
            # If not within TruChain._call, just wrap original method as is.
            if self.record is None:
                return func(*args, **kwargs)

            else:
                # Otherwise keep track of inputs and outputs (or exception).

                error = None
                ret = None

                start_time = datetime.now()

                try:
                    # Using sig bind here so we can produce a list of key-value
                    # pairs even if positional arguments were provided.
                    bindings: BoundArguments = sig.bind(*args, **kwargs)
                    ret = func(*bindings.args, **bindings.kwargs)
                    
                except BaseException as e:
                    error = e

                end_time = datetime.now()

                # Don't include self in the recorded arguments.
                nonself = {k: v for k, v in bindings.arguments.items() if k != "self"}
                row = ChainCall(input=nonself, start_time=start_time, end_time=end_time)

                if error is not None:
                    row = replace(row, error = error)
                else:
                    row = replace(row, output = ret)

                key = self._address_hashable(address)
                self.record[key].append(row)
                
                if error is not None:
                    raise error

                return ret
        
        wrapper._instrumented = func

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
            # superchains, and we want to capture the intermediate steps

            if "langchain" not in str(cp):
                continue

            if hasattr(cp, "_call"):
                original_fun = getattr(cp, "_call")

                if self.verbose:
                    print(f"instrumented {cp}._call")

                setattr(cp, "_call", self._instrument_method(address, original_fun))

            if hasattr(cp, "_chain_type"):
                prop = getattr(cp, "_chain_type")
                setattr(cp, "_chain_type", self._instrument_chain_type(chain=chain, prop=prop))

        # Not using chain.dict() here as that recursively converts subchains to
        # dicts but we want to capture their class information here.
        for k in chain.__fields__:
            v = getattr(chain, k)
            if isinstance(v, str):
                pass
            elif isinstance(v, Chain):
                self._instrument(v, address=address + (k, ))
            elif isinstance(v, Sequence):
                for i, sv in enumerate(v):
                    if isinstance(sv, Chain):
                        self._instrument(sv, address=address + (k, i, ))
                

        # TODO: check if we want to capture anything not accessible through __fields__ .
