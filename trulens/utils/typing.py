"""Type aliases and data container classes. """

# NOTE(piotrm) had to move some things to this file to avoid circular
# dependencies between the files that originally contained those things.

from dataclasses import dataclass, field
from inspect import signature
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Tuple, TypeVar, Union)

import numpy as np
from trulens.nn.backend import Tensor

# Atomic model inputs (at least from our perspective)
DataLike = Union[np.ndarray, Tensor]

# Model input arguments. Either a single element or list/tuple of several.
ArgsLike = Union[DataLike, List[DataLike], Tuple[DataLike]]

# Model input kwargs.
Kwargs = Dict[str, DataLike]

# Some backends allow inputs in terms of dicts with tensors as keys (i.e. feed_dict in tf1).
# We keep track of internal names so that these feed dicts can use strings as names instead.
Feed = Dict[Union[str, Tensor], DataLike]

KwargsLike = Union[Kwargs, Feed]

C = TypeVar("C")
C1 = TypeVar("C1")
C2 = TypeVar("C2")
K = TypeVar("K")
V = TypeVar("V")

Indexable = Union[List[V], Tuple[V]]
# For type checking the above.
DATA_CONTAINER_TYPE = (list, tuple)

# Convert a single element input to the multi-input one.
def as_args(ele):
    if isinstance(ele, DATA_CONTAINER_TYPE):
        return ele
    else:
        return [ele]

def replace(l: Indexable[V], i: int, v: V) -> Indexable[V]:
    """Copy of the given list or tuple with the given index replaced by the given value."""
    if isinstance(l, list):
        l = l.copy()
        l[i] = v
    elif isinstance(l, tuple):
        l = list(l)
        l[i] = v
        l = tuple(l)
    else:
        raise ValueError(f"list or tuple expected but got {l.__class__.__name__}")

    return l

def replace_dict(d: Dict[K, V], k: K, v: V) -> Dict[K, V]:
    """Copy of the given dictionary with the given key replaced by the given value."""
    d = d.copy()
    d[k] = v
    return d

def iter_then(iter1: Iterable[V], iter2: Iterable[V]) -> Iterable[V]:
    """Iterate through the given iterators, one after the other."""
    for x in iter1:
        yield x
    for x in iter2:
        yield x

@dataclass
class Lens(Generic[C, V]):
    """Simple lenses implementation."""

    get: Callable[[C], V]
    set: Callable[[C, V], C]

    @staticmethod
    def lenses_elements(c: List[V]) -> Iterable['Lens[List[V], V]']:
        """Lenses focusing on elements of a list."""
        for i in range(len(c)):
            yield Lens(lambda l, i=i: l[i], lambda l, v,i=i: replace(l, i, v))

    @staticmethod
    def lenses_values(c: Dict[K, V]) -> Iterable['Lens[Dict[K, V], V]']:
        """Lenses focusing on values in a dictionary."""
        for k in c.keys():
            yield Lens(get=lambda d, k=k: d[k], set=lambda d, v, k=k: replace_dict(d, k, v))

    @staticmethod
    def compose(l1: 'Lens[C1, C2]', l2: 'Lens[C2, V]') -> 'Lens[C1, V]':
        """Compose two lenses."""
        return Lens(
            lambda c: l2.get(l1.get(c)),
            lambda c, e: l1.set(c, l2.set(l1.get(c), e))
        )

@dataclass
class ModelInputs:
    """Container for model input arguments, that is, args and kwargs."""

    args: List[DataLike] = field(default_factory=list)
    kwargs: KwargsLike = field(default_factory=dict)

    lens_args: Lens['ModelInputs', DataLike] = Lens(lambda s: s.args, lambda s, a: ModelInputs(a, s.kwargs))
    lens_kwargs = Lens(lambda s: s.kwargs, lambda s, kw: ModelInputs(s.args, kw))

    def __init__(self, args: ArgsLike, kwargs: KwargsLike):
        self.args = as_args(args)
        self.kwargs = kwargs

    def lenses_values(self):
        return iter_then(
            (Lens.compose(ModelInputs.lens_args, l) for l in Lens.lenses_elements(self.args)),
            (Lens.compose(ModelInputs.lens_kwargs, l) for l in Lens.lenses_values(self.kwargs))
        )

    def values(self):
        for l in self.lenses_values():
            yield l.get(self)

    def map(self, f):
        ret = self
        for l in self.lenses_values():
            ret = l.set(ret, f(l.get(ret)))
        return ret

    def foreach(self, f):
        for l in self.lenses_values():
            f(l.get(self))

    def first(self):
        try:
            return next(self.values())
        except StopIteration:
            raise ValueError("ModelInputs is without arguments nor keyword arguments")


def accepts_model_inputs(func: Callable) -> bool:
    """Determine whether the given callable takes in model inputs or just
    activations."""

    return "model_inputs" in signature(func).parameters


# Baselines are either explicit or computable from the same data as sent to DoI
# __call__ .
BaselineLike = Union[ArgsLike, Callable[[ArgsLike, Optional[ModelInputs]],
                                         ArgsLike]]

# Interventions for fprop specifiy either activations at some non-InputCut or
# model inputs if DoI is InputCut (these include both args and kwargs).
# Additionally, some backends (tf1) provie interventions as kwargs instead.
InterventionLike = Union[ArgsLike, KwargsLike, ModelInputs]
