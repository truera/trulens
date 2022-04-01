"""Type aliases and data container classes. 

Large part of these utilities are meant to support a design decision: public
methods are permissive in the types they accept while private methods are not. 

For example, 

    attributions(*model_args: ArgsLike, **model_kwargs: ModelKwargs)

For models that take in a single input, model_args can be a single Tensor or
numpy array. For models that take in multiple inputs, model_args must be some
iterable over Tensors or numpy arrays. The implication is that the user can call
this method as in "attributions(inputA)" instead of attributions([inputA]). The
other permissibility is that both backend Tensors and numpy arrays are supported
and the method returns results in the same type. 

List of flexible typings:

    - Single vs. Multiple Inputs -- Can pass in a single value instad of list.
      This is indicated by the `ArgsLike` type alias (paraphrased):

      ``` ArgsLike[V] = Union[
        V, Inputs[V] # len != 1 
      ]
      ```

        - AttributionMethod.attributions: model_args argument
        - DoI.__call__: z argument
        - QoI.__call__: y argument
        - LinearDoI.__init__: baseline argument
        - LinearDoI.baseline (when callable): first argument
        - ModelWrapper.fprop: model_args, intervention arguments
        - ModelWrapper.qoi_bprop: model_args argument

    - Single vs. Multiple QoI outputs -- A quantity of interest may return one
      or more tensors.

        - QoI.__call__ return

    - Backend Tensor vs. Numpy Array -- Can pass in a backend Tensor or numpy
      array. This is indicated by the `DataLike` type alias:

      ``` DataLike = Union[np.ndarray, Tensor] ```

        - AttributionMethod.attributions: model_args, model_kwargs arguments
          (return type based first arg in either).
        - DoI.__call__: z and model_inputs arguments (return type based on z).
        - QoI.__call__: y argument. Note however that QoI must return Tensor(s).
        - LinearDoI.baseline (when callable): first argument
        - ModelWrapper.fprop: model_args, model_kwargs, intervention arguments
        - ModelWrapper.qoi_bprop: model_args, model_kwargs, intervention
          arguments.

      Of the various flexibilities for the public mentioned here, this is the
      only one that is also maintained in the private methods.

    - ModelInputs input -- Can define method to accept ModelInputs structure or
      just the args. To indicate the latter, must be keyword argument named
      "model_inputs".

        - DoI.__call__
        - LinearDoI.baseline (when callable): see BaselineLike.

    - ModelInputs or args -- Can pass in single value, iterable, dictionary, or
      ModelInputs structure. These correspond to single positional arg, multiple
      positional args, keyword args, and (both positional and keyword args).
    
        - ModelWrapper.fprop: intervention argument, see InterventionLike (paraphrased):

        ``` InterventionLike = Union[
              ArgsLike[DataLike], KwargsLike, ModelInputs]
        ```

        - ModelWrapper.qoi_bprop: intervention argument: see InterventionLike

    - Callable Baselines -- Baselines can be methods that return baseline
      values. See (paraphrased) BaselineLike (paraphrased):

      ``` BaselineLike = Union[
          ArgsLike[DataLike], ArgsLike[DataLike] -> ArgsLike[DataLike],
          (ArgsLike[DataLike], ModelInputs) -> ArgsLike[DataLike]
      ]
      ```

        - LinearDoI.__init__: baseline argument

Part of the implementation of this design are private variants of public methods
that do not have most of the above flexibility. Most, however, have the tensor
vs. numpy array flexibility. These are to be used within Truera for more
consistent expectations regarding inputs/outputs.

    - DoI._wrap_public_call vs. DoI.__call__
    - DoI._wrap_public_get_activation_multiplier vs DoI.get_activation_multiplier
    - QoI._wrap_public_call vs. QoI.__call__
    - ModelWrapper._fprop vs. ModelWrapper.fprop
    - ModelWrapper._qoi_bprop vs. ModelWrapper.qoi_bprop
"""

# NOTE(piotrm) had to move some things to this file to avoid circular
# dependencies between the files that originally contained those things.

from dataclasses import dataclass
from dataclasses import field
from inspect import signature
from typing import (
    Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, Type,
    TypeVar, Union
)

import numpy as np

# C for "container"
C = TypeVar("C")
C1 = TypeVar("C1")
C2 = TypeVar("C2")
# K for "key"
K = TypeVar("K")
# V for "value"
V = TypeVar("V")
# Another, different value
U = TypeVar("U")

# Lists that represent the potentially multiple inputs to some neural layer.
Inputs = List

# Lists that represent model outputs or quantities of interest if multiple.
Outputs = List

# Lists that are meant to represent instances of some uniform distribution.
Uniform = List

# "One or More" OM[C, V] (V for Value, C for Container)
OM = Union[V, C]  # actually C[V] but cannot get python to accept that
# e.g. OM[List, DataLike] - One or more DataLike where the more is contained in a List
# e.g. OM[Inputs, DataLike] - One or more DataLike that represent model inputs.

# "One or More" where the container type is itself a one or more.
OMNested = Union[OM[V, C], 'OMNested[C, V]']

# Each backend should define this.
Tensor = TypeVar("Tensor")

ModelLike = Union['tf.Graph',  # tf1 
                  'keras.Model',  # keras
                  'tensorflow.keras.Model',  # tf2
                  'torch.nn.Module',  # pytorch
                 ]

# Atomic model inputs (at least from our perspective)
DataLike = Union[np.ndarray, Tensor]

# Model input arguments. Either a single element or list/tuple of several.
# User-provided methods are given DataLike and for single input models/layers
# but internally we pass around InputList[DataLike] for consistency. Likewise
# user-provided methods may return a single DataLike or Uniform[DataLike] but we
# want to pass around Inputs[DataLike] or Inputs[Uniform[DataLike]]. Because of
# this, there are checks with wrapping/unwrapping around user-provided-function
# calls using the various utilities below. Purely internal methods should not be
# doing wrapping/unwrapping. Leaving this type only for the public methods while
# the private methods use the more informative OM type with purpose annotation.
ArgsLike = OM[Union[List, Tuple], V]  # ArgsLike[V]

# Model input kwargs.
Kwargs = Dict[str, V]  # Kwargs[V]

# Some backends allow inputs in terms of dicts with tensors as keys (i.e. feed_dict in tf1).
# We keep track of internal names so that these feed dicts can use strings as names instead.
Feed = Dict[Union[str, Tensor], DataLike]

KwargsLike = Union[Kwargs[DataLike], Feed]

Indexable = Union[List[V], Tuple[V]]  # Indexable[V]
# For checking the above against an instance:
DATA_CONTAINER_TYPE = (list, tuple)

## Utilities for dealing with nested structures


def nested_map(y: OMNested[C, U], fn: Callable[[U], V]) -> OMNested[C, V]:
    """
    Applies fn to non-container elements in y. This works on "one or more" and even mested om.

    Parameters
    ----------
    y:  non-collective object or a nested list/tuple of objects
        The leaf objects will be inputs to fn.
    fn: function
        The function applied to leaf objects in y. Should take in a single
        non-collective object and return a non-collective object.
    Returns
    ------
    non-collective object or a nested list or tuple
        Has the same structure as y, and contains the results of fn applied to
        leaf objects in y.

    """
    if isinstance(y, DATA_CONTAINER_TYPE):
        out = []
        for i in range(len(y)):
            out.append(nested_map(y[i], fn))
        return y.__class__(out)
    else:
        return fn(y)


def nested_cast(
    backend: 'Backend',
    args: OMNested[C, DataLike],
    astype: Type  # : select one of the two types of DataLike
) -> Union[OMNested[C, Tensor], OMNested[C, np.ndarray]]:  # : of selected type
    """Transform set of values to the given type wrapping around List/Tuple if
    needed."""

    caster = datalike_caster(backend, astype)

    return nested_map(args, lambda x: caster(x) if type(x) != astype else x)


def tab(s, tab="  "):
    return "\n".join(map(lambda ss: tab + ss, s.split("\n")))


def nested_str(items: OMNested[C, DataLike]) -> str:
    ret = ""

    if isinstance(items, DATA_CONTAINER_TYPE):
        ret = f"[{len(items)}\n"
        inner = "\n".join(map(nested_str, items))
        ret += tab(inner) + "\n]"

    else:
        if hasattr(items, "shape"):
            ret += f"{items.__class__.__name__} {items.dtype} {items.shape}"
        else:
            ret += str(items)

    return ret


## Utilities for dealing with DataLike


def datalike_caster(backend: 'Backend',
                    astype: Type) -> Callable[[DataLike], DataLike]:
    """Return a method that lets one cast a DataLike to the specified type by
    the given backend."""

    if issubclass(astype, np.ndarray):
        caster = backend.as_array
    elif issubclass(astype, backend.Tensor):
        caster = backend.as_tensor
    else:
        raise ValueError(f"Cannot cast unhandled type {astype}.")

    return caster


## Utilities for dealing with "one or more"


def om_assert_matched_pair(a1, a2):
    """
    Assert that two "one or more"'s are of the same type, contain the same number
    of elements (if containers) and that those elements are also of the same
    types."""

    assert type(a1) == type(
        a2
    ), f"OM's are of different types: {type(a1)} vs {type(a2)}"

    if isinstance(a1, DATA_CONTAINER_TYPE):
        assert len(a1) == len(
            a2
        ), f"OM's are of different lengths: {len(a1)} vs {len(a2)}"

        for i, (a, b) in enumerate(zip(a1, a2)):
            assert type(a) == type(
                b
            ), f"OM's elements {i} are of different types: {type(a)} vs {type(b)}"


def many_of_om(args: OM[C, V], innertype: Type = None) -> 'C[V]':
    """
    Convert "one or more" (possibly a single V/DataLike) to List/C[V]. For cases
    where the element type V is also expected to be a container, provide the
    innertype argument so that the check here can be done properly.

    Opposite of `om_of_many`.
    """
    if isinstance(args, DATA_CONTAINER_TYPE) and (innertype is None or
                                                  len(args) == 0 or isinstance(
                                                      args[0], innertype)):
        return args
    elif innertype is None or isinstance(args, innertype):
        return [args]  # want C(args) but cannot materialize C in python
    else:
        raise ValueError(f"Unhandled One-Or-More type {type(args)}")


def om_of_many(inputs: 'C[V]') -> OM[C, V]:
    """
    If there is more than 1 thing in container, will remain a container,
    otherwise will become V (typically DataLike).
    
    Opposite of `many_of_om`.
    """

    if isinstance(inputs, DATA_CONTAINER_TYPE):
        if len(inputs) == 1:
            return inputs[0]
        else:
            return inputs
    else:
        return inputs


## Other utilities


class IndexableUtils:

    def with_(l: Indexable[V], i: int, v: V) -> Indexable[V]:
        """Copy of the given list or tuple with the given index replaced by the
        given value."""

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

    args: Inputs[DataLike] = field(default_factory=list)
    kwargs: KwargsLike = field(default_factory=dict)

    lens_args: Lens['ModelInputs', Inputs[DataLike]] = Lens(
        lambda s: s.args, lambda s, a: ModelInputs(a, s.kwargs)
    )
    # lens focusing on the args field of this container.

    lens_kwargs: Lens['ModelInputs', KwargsLike] = Lens(
        lambda s: s.kwargs, lambda s, kw: ModelInputs(s.args, kw)
    )

    # lens focusing on the kwargs field of this container.

    def __init__(self, args: Inputs[DataLike] = [], kwargs: KwargsLike = {}):
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
BaselineLike = Union[ArgsLike[DataLike],
                     Callable[[ArgsLike[DataLike], Optional[ModelInputs]],
                              ArgsLike[DataLike]]]

# Interventions for fprop specifiy either activations at some non-InputCut or
# model inputs if DoI is InputCut (these include both args and kwargs).
# Additionally, some backends (tf1) provide interventions as kwargs instead.
InterventionLike = Union[ArgsLike[DataLike], KwargsLike, ModelInputs]


def float_size(name: str) -> int:
    """Given a name of a floating type, guess its size in bytes."""

    # TODO: Do this better.

    # Different backends have their own type structures so this is hard to
    # generalize. Just guessing based on bitlengths in names for now.

    if name == "double":
        name = "float64"

    if "float" not in name:
        raise ValueError("Type name {name} does not refer to a float.")

    if name == "float":
        return 4

    if "128" in name:
        return 16
    elif "64" in name:
        return 8
    elif "32" in name:
        return 4
    elif "16" in name:
        return 2

    raise ValueError(f"Cannot guess size of {name}")
