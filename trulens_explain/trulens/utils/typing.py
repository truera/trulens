"""
Type aliases and data container classes. 

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
      array. This is indicated by the `TensorLike` type alias:

      ``` TensorLike = Union[np.ndarray, Tensor] ```

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
      a Tensors (includes ModelInputs) structure. These correspond to single
      positional arg, multiple positional args, keyword args, and (both
      positional and keyword args).
    
        - ModelWrapper.fprop: intervention argument, see InterventionLike
          (paraphrased):

        ``` InterventionLike = Union[
              ArgsLike[TensorLike], KwargsLike, Tensors]
        ```

        - ModelWrapper.qoi_bprop: intervention argument: see InterventionLike

    - Callable Baselines -- Baselines can be methods that return baseline
      values. See (paraphrased) BaselineLike (paraphrased):

      ``` BaselineLike = Union[
          ArgsLike[TensorLike], ArgsLike[TensorLike] -> ArgsLike[TensorLike],
          (ArgsLike[TensorLike], ModelInputs) -> ArgsLike[TensorLike]
      ]
      ```

        - LinearDoI.__init__: baseline argument

Part of the implementation of this design are private variants of public methods
that do not have most of the above flexibility. Most, however, have the tensor
vs. numpy array flexibility. These are to be used within Truera for more
consistent expectations regarding inputs/outputs.

    - DoI._wrap_public_call vs. DoI.__call__
    - DoI._wrap_public_get_activation_multiplier vs
      DoI.get_activation_multiplier
    - QoI._wrap_public_call vs. QoI.__call__
    - ModelWrapper._fprop vs. ModelWrapper.fprop
    - ModelWrapper._qoi_bprop vs. ModelWrapper.qoi_bprop

Dealing with Magic Numbers for Axes Indices

    Some multi-dimensional (or nested) containers contain type hint and
    annotations for order of axes. For example AttributionResults:

    AttributionResult.axes == 

    {'attributions': [
        trulens.utils.typing.Outputs,
        trulens.utils.typing.Inputs,
        typing.Union[numpy.ndarray, ~Tensor] # == TensorLike
     ], 
     'gradients': [
         trulens.utils.typing.Outputs,
         trulens.utils.typing.Inputs,
         trulens.utils.typing.Uniform,
         typing.Union[numpy.ndarray, ~Tensor] # == TensorLike
     ],
     'interventions': [
         trulens.utils.typing.Inputs,
         trulens.utils.typing.Uniform,
         typing.Union[numpy.ndarray, ~Tensor] # == TensorLike
     ]}

     You can then lookup an axis of interest:

     - gradients_axes = AttributionResult.axes['gradients']
     - gradients_axes.index(Outputs) == 0 
     - gradients_axes.index(TensorLike) == 3

"""

# NOTE(piotrm) had to move some things to this file to avoid circular
# dependencies between the files that originally contained those things.

from abc import ABC
from abc import abstractmethod
import collections
from dataclasses import dataclass
from dataclasses import field
from inspect import signature
from typing import (
    Callable, Dict, Generic, Iterable, List, Optional, Tuple, Type, TypeVar,
    Union
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


# Lists that represent multiple inputs to some neural layer.
class Inputs(Generic[V], List[V]):
    ...


# Lists that represent model outputs or quantities of interest if multiple.
class Outputs(Generic[V], List[V]):
    ...


# Lists that represent instances of a uniform distribution.
class Uniform(Generic[V], List[V]):
    ...


# "One or More" OM[C, V] (V for Value, C for Container)
OM = Union[V, C]  # actually C[V] but cannot get python to accept that
# - e.g. OM[List, TensorLike] - One or more TensorLike where the more is
#   contained in a List
# - e.g. OM[Inputs, TensorLike] - One or more TensorLike that represent model
#   inputs.

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
TensorLike = Union[np.ndarray, Tensor]

# Model input arguments. Either a single element or list/tuple of several.
# User-provided methods are given TensorLike and for single input models/layers
# but internally we pass around InputList[TensorLike] for consistency. Likewise
# user-provided methods may return a single TensorLike or Uniform[TensorLike]
# but we want to pass around Inputs[TensorLike] or Inputs[Uniform[TensorLike]].
# Because of this, there are checks with wrapping/unwrapping around
# user-provided-function calls using the various utilities below. Purely
# internal methods should not be doing wrapping/unwrapping. Leaving this type
# only for the public methods while the private methods use the more informative
# OM type with purpose annotation.
ArgsLike = OM[Union[List, Tuple], V]  # ArgsLike[V]

# Model input kwargs.
Kwargs = Dict[str, V]  # Kwargs[V]

# Some backends allow inputs in terms of dicts with tensors as keys (i.e.
# feed_dict in tf1). We keep track of internal names so that these feed dicts
# can use strings as names instead.
Feed = Dict[Union[str, Tensor], TensorLike]

KwargsLike = Union[Kwargs[TensorLike], Feed]

Indexable = Union[List[V], Tuple[V]]  # Indexable[V]
# For checking the above against an instance:
DATA_CONTAINER_TYPE = (list, tuple, Outputs, Inputs, Uniform)
MAP_CONTAINER_TYPE = (collections.abc.Mapping,)
## Utilities for dealing with nested structures


def nested_axes(typ):
    """
    Given a type annotation containing a nested structure of single argument
    types, return a list of the nested types in order from outer to inner. Stop
    at TensorLike.
    """

    if typ == TensorLike:
        return [TensorLike]

    args = typ.__args__

    if len(args) != 1:
        raise ValueError(
            f"Cannot extract axis order from multi-argument type {typ}."
        )

    ret = [typ.__origin__] + nested_axes(args[0])

    return ret


def numpy_of_nested(backend, x: OMNested[Iterable, TensorLike]) -> np.ndarray:
    """
    Convert arbitrarily nested tensors into a numpy ndarray. This likely
    requires uniform dimensions of all items on the same level of nesting.
    """

    x = nested_cast(backend=backend, astype=np.ndarray, args=x)

    return np.array(x)


def nested_map(
    y: OMNested[C, U],
    fn: Callable[[U], V],
    *,
    check_accessor: Callable[[C], V] = None,
    nest: int = 999
) -> OMNested[C, V]:
    """
    Applies fn to non-container elements in y. This works on "one or more" and
    even mested om.

    Parameters
    ----------
    y:  non-collective object or a nested list/tuple of objects
        The leaf objects will be inputs to fn.
    fn: function
        The function applied to leaf objects in y. Should take in a single
        non-collective object and return a non-collective object.
    check_accessor: function
        A way to make instance checks from the container level.
    nest: int
        Another way to specify which level to apply the function. This is the only way to apply a fn on a DATA_CONTAINER_TYPE. 
        Currently MAP_CONTAINER_TYPE is not included in the nesting levels as they usually wrap tensors and functionally are not an actual container.
    Returns
    ------
    non-collective object or a nested list or tuple
        Has the same structure as y, and contains the results of fn applied to
        leaf objects in y.

    """
    if check_accessor is not None:
        try:
            check_y = check_accessor(y)
            if not isinstance(check_y,
                              DATA_CONTAINER_TYPE + MAP_CONTAINER_TYPE):
                return fn(y)
        except:
            pass
    if isinstance(y, DATA_CONTAINER_TYPE) and nest > 0:
        out = []
        for i in range(len(y)):
            out.append(
                nested_map(
                    y[i], fn, check_accessor=check_accessor, nest=nest - 1
                )
            )
        return y.__class__(out)
    if isinstance(y, MAP_CONTAINER_TYPE):
        out = {}
        for k in y.keys():
            out[k] = nested_map(
                y[k], fn, check_accessor=check_accessor, nest=nest
            )
        return y.__class__(out)
    else:
        return fn(y)


def nested_zip(y1: OMNested[C, U],
               y2: OMNested[C, V],
               nest=999) -> OMNested[C, Tuple[U, V]]:
    """
    zips at the element level each element in y1 witheach element in y2. This works on "one or more" and even mested om.

    Parameters
    ----------
    y1:  non-collective object or a nested list/tuple of objects
        The leaf objects will be zipped.
    y2:  non-collective object or a nested list/tuple of objects
        The leaf objects will be zipped.
    nest: int
        A way to specify which level to apply the zip. This is the only way to apply a zip on a DATA_CONTAINER_TYPE. 
        Currently MAP_CONTAINER_TYPE is not included in the nesting levels as they usually wrap tensors and functionally are not an actual container.
    Returns
    ------
    non-collective object or a nested list or tuple
        Has the same structure as y1, zips leaf objects in y1 with leaf objects in y2.

    """
    if isinstance(y1, DATA_CONTAINER_TYPE) and nest > 0:
        assert len(y1) == len(y2)
        out = []
        for i in range(len(y1)):
            out.append(nested_zip(y1[i], y2[i], nest - 1))
        return y1.__class__(out)
    if isinstance(y1, MAP_CONTAINER_TYPE):
        out = {}
        assert len(y1.keys()) == len(y2.keys())
        for k in y1.keys():
            out[k] = nested_zip(y1[k], y2[k], nest)
        return y1.__class__(out)
    else:
        return (y1, y2)


def nested_cast(
    backend: 'Backend',
    args: OMNested[C, TensorLike],
    astype: Type  # : select one of the two types of TensorLike
) -> Union[OMNested[C, Tensor], OMNested[C, np.ndarray]]:  # : of selected type
    """Transform set of values to the given type wrapping around List/Tuple if
    needed."""

    caster = TensorLike_caster(backend, astype)

    return nested_map(
        args, lambda x: caster(x) if not issubclass(type(x), astype) else x
    )


def tab(s: str, tab: str = "  "):
    """
    Prepend `tab` to each line of input.
    """

    return "\n".join(map(lambda ss: tab + ss, s.split("\n")))


def nested_str(items: OMNested[C, TensorLike]) -> str:
    """
    Generate string with shape information of the contents of given `items`.
    """

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


## Utilities for dealing with TensorLike


def TensorLike_caster(backend: 'Backend',
                      astype: Type) -> Callable[[TensorLike], TensorLike]:
    """Return a method that lets one cast a TensorLike to the specified type by
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
    Convert "one or more" (possibly a single V/TensorLike) to List/C[V]. For cases
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
    otherwise will become V (typically TensorLike).
    
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


class Tensors(ABC):
    """Model inputs or internal layer inputs."""

    @abstractmethod
    def as_model_inputs(self) -> 'ModelInputs':
        pass


@dataclass
class TensorAKs(Tensors):  # "Tensor Args and Kwargs"
    """Container for positional and keyword arguments."""

    args: ArgsLike = field(default_factory=list)
    kwargs: KwargsLike = field(default_factory=dict)

    # lens focusing on the args field of this container.
    lens_args: Lens[
        'TensorAKs',
        ArgsLike] = Lens(lambda s: s.args, lambda s, a: TensorAKs(a, s.kwargs))

    # lens focusing on the kwargs field of this container.
    lens_kwargs: Lens['TensorAKs', KwargsLike] = Lens(
        lambda s: s.kwargs, lambda s, kw: TensorAKs(s.args, kw)
    )

    def __init__(self, args: List[TensorLike] = [], kwargs: KwargsLike = {}):
        self.args = args
        self.kwargs = kwargs

        # Check to make sure we don't accidentally set args to a single element.
        if not isinstance(args, DATA_CONTAINER_TYPE):
            raise ValueError(f"container expected but is {args.__class__}")

    def __len__(self):
        return len(self.args) + len(self.kwargs)

    def __contains__(self, item):
        for val in self.values():
            if item is val:
                return True
        return False

    def lenses_values(self):
        """Get lenses focusing on each contained value."""

        return IterableUtils.then_(
            (
                Lens.compose(TensorAKs.lens_args, l)
                for l in Lens.lenses_elements(self.args)
            ), (
                Lens.compose(TensorAKs.lens_kwargs, l)
                for l in Lens.lenses_values(self.kwargs)
            )
        )

    def values(self):
        """Get the contained values."""

        for l in self.lenses_values():
            yield l.get(self)

    def map(self, f):
        """Produce a new set of args by transforming each value with the given function."""
        new_args = nested_map(self.args, f)
        new_kwargs = nested_map(self.kwargs, f)
        return TensorAKs(args=new_args, kwargs=new_kwargs)

    def foreach(self, f):
        """Apply the given function to each value."""

        for l in self.lenses_values():
            nested_map(l.get(self), f)

    def first_batchable(self, backend):
        """Find the first object that may be considered a batchable input."""

        for v in self.values():
            # Current assumption is that batchable items are in the args list or kwargs vals. Continuing this assumption until other use cases seen.
            if isinstance(v, (np.ndarray, backend.Tensor)):
                return v
            # However, we will look one level deep if a dict is found: there is a known case where placeholders expect tensor inputs as dict.
            elif isinstance(v, dict):
                for k in v.keys():
                    if isinstance(v[k], (np.ndarray, backend.Tensor)):
                        return v[k]
        return None

    def call_on(self, f):
        """Call the given method with the contained arguments."""

        return f(*self.args, **self.kwargs)

    def as_model_inputs(self) -> 'ModelInputs':
        return ModelInputs(self.args, self.kwargs)


class TensorArgs(TensorAKs):
    """Container for layer input that contain only positional arguments.
    Presently internal layers only accept positional arguments; keyword
    arguments are only for model inputs."""

    def __init__(self, args: List[TensorLike]):
        super(TensorArgs, self).__init__(args, {})


class ModelInputs(TensorAKs):
    """
    Positional and keyword arguments meant for model inputs.

    This is a separate class instead of alias to be able to do isinstance checks
    with it.
    """


def accepts_model_inputs(func: Callable) -> bool:
    """Determine whether the given callable takes in model inputs or just
    activations."""

    return "model_inputs" in signature(func).parameters


# Baselines are either explicit or computable from the same data as sent to DoI
# __call__ .
BaselineLike = Union[ArgsLike[TensorLike],
                     Callable[[ArgsLike[TensorLike], Optional[ModelInputs]],
                              ArgsLike[TensorLike]]]

# Interventions for fprop specifiy either activations at some non-InputCut or
# model inputs if DoI is InputCut (these include both args and kwargs).
# Additionally, some backends (tf1) provide interventions as kwargs instead.
InterventionLike = Union[ArgsLike[TensorLike], KwargsLike, Tensors]


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


def float_size(name: str) -> int:
    """Given a name of a floating type, guess its size in bytes."""

    # TODO: Do this better. Different backends have their own type structures so
    # this is hard to generalize. Just guessing based on bitlengths in names for
    # now.

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
