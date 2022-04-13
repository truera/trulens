"""Type aliases and data container classes. """

# NOTE(piotrm) had to move some things to this file to avoid circular
# dependencies between the files that originally contained those things.

from dataclasses import dataclass
from dataclasses import field
from inspect import signature
from typing import (
    Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar,
    Union
)

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

# C for "container"
C = TypeVar("C")
C1 = TypeVar("C1")
C2 = TypeVar("C2")
# K for "key"
K = TypeVar("K")
# V for "value"
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


class IndexableUtils:

    def with_(l: Indexable[V], i: int, v: V) -> Indexable[V]:
        """Copy of the given list or tuple with the given index replaced by the given value."""

        if isinstance(l, list):
            l = l.copy()
            l[i] = v
        elif isinstance(l, tuple):
            l = list(l)
            l[i] = v
            l = tuple(l)
        else:
            raise ValueError(
                f"list or tuple expected but got {l.__class__.__name__}"
            )

        return l


class IterableUtils:

    def then_(iter1: Iterable[V], iter2: Iterable[V]) -> Iterable[V]:
        """Iterate through the given iterators, one after the other."""

        for x in iter1:
            yield x
        for x in iter2:
            yield x


class DictUtils:

    def with_(d: Dict[K, V], k: K, v: V) -> Dict[K, V]:
        """Copy of the given dictionary with the given key replaced by the given value."""

        d = d.copy()
        d[k] = v
        return d


@dataclass
class Lens(Generic[C, V]):  # Container C with values V
    """
    Simple lenses implementation. Lenses are a common paradigm for dealing with
    data structures in a functional manner. More info here:
    https://docs.racket-lang.org/lens/lens-intro.html#%28tech._lens%29 .
    Presently we only use lenses to unify access/update to the positional and
    keyword parameters in ModelView.
    """

    get: Callable[[C], V]
    # Given a container, extract the focused value.

    set: Callable[[C, V], C]
    # Given a container and a value, replace the focused value with the new one,
    # returning a new container.

    @staticmethod
    def lenses_elements(c: List[V]) -> Iterable['Lens[List[V], V]']:
        """Lenses focusing on elements of a list."""

        for i in range(len(c)):
            yield Lens(
                lambda l, i=i: l[i],
                lambda l, v, i=i: IndexableUtils.with_(l, i, v)
            )

    @staticmethod
    def lenses_values(c: Dict[K, V]) -> Iterable['Lens[Dict[K, V], V]']:
        """Lenses focusing on values in a dictionary."""

        for k in c.keys():
            yield Lens(
                get=lambda d, k=k: d[k],
                set=lambda d, v, k=k: DictUtils.with_(d, k, v)
            )

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

    lens_args: Lens['ModelInputs', List[DataLike]] = Lens(
        lambda s: s.args, lambda s, a: ModelInputs(a, s.kwargs)
    )
    # lens focusing on the args field of this container.

    lens_kwargs: Lens['ModelInputs', KwargsLike] = Lens(
        lambda s: s.kwargs, lambda s, kw: ModelInputs(s.args, kw)
    )

    # lens focusing on the kwargs field of this container.

    def __init__(self, args: List[DataLike] = [], kwargs: KwargsLike = {}):
        """
        Contain positional and keyword arguments. This should operate when given
        args received from a method with this signature:

        def method(*args, **kwargs):
            something = ModelInputs(args, kwargs)

        method([arg1, arg2]) method(arg1, arg2)
        
        In this first case, *args is a tuple with one element whereas in the
        second, it is a tuple with two. In either case, we convert it to a list
        with two.
        """

        if isinstance(args, tuple) and len(args) == 1 and isinstance(
                args[0], DATA_CONTAINER_TYPE):
            args = args[0]

        self.args = args
        self.kwargs = kwargs

        if not isinstance(args, DATA_CONTAINER_TYPE):
            raise ValueError(f"container expected but is {args.__class__}")

    def __len__(self):
        return len(self.args) + len(self.kwargs)

    def lenses_values(self):
        """Get lenses focusing on each contained value."""

        return IterableUtils.then_(
            (
                Lens.compose(ModelInputs.lens_args, l)
                for l in Lens.lenses_elements(self.args)
            ), (
                Lens.compose(ModelInputs.lens_kwargs, l)
                for l in Lens.lenses_values(self.kwargs)
            )
        )

    def values(self):
        """Get the contained values."""

        for l in self.lenses_values():
            yield l.get(self)

    def map(self, f):
        """Produce a new set of inputs by transforming each value with the given function."""

        ret = self
        for l in self.lenses_values():
            ret = l.set(ret, f(l.get(ret)))
        return ret

    def foreach(self, f):
        """Apply the given function to each value."""

        for l in self.lenses_values():
            f(l.get(self))

    def first(self):
        """Get the first value, whether it is an arg or kwargs. args come first."""

        try:
            return next(self.values())
        except StopIteration:
            raise ValueError(
                "ModelInputs had neither arguments nor keyword arguments."
            )

    def call_on(self, f):
        """Call the given method with the contained arguments."""

        return f(*self.args, **self.kwargs)


def accepts_model_inputs(func: Callable) -> bool:
    """Determine whether the given callable takes in model inputs or just
    activations."""

    return "model_inputs" in signature(func).parameters


# Baselines are either explicit or computable from the same data as sent to DoI
# __call__ .
BaselineLike = Union[DataLike, Callable[[ArgsLike, Optional[ModelInputs]],
                                        ArgsLike]]

# Interventions for fprop specifiy either activations at some non-InputCut or
# model inputs if DoI is InputCut (these include both args and kwargs).
# Additionally, some backends (tf1) provide interventions as kwargs instead.
InterventionLike = Union[DataLike, KwargsLike, ModelInputs]

def render_object(obj, keys=None):
    """Render an instance of some class in a concise manner."""

    temp = obj.__class__.__name__ + "("

    if keys is None:
        keys = dir(obj)

    vals = []
    for k in keys:
        vals.append(f"{k}={getattr(obj, k)}")

    temp += ",".join(vals) + ")"

    return temp
