"""Type aliases and data container classes. """

# NOTE(piotrm) had to move some things to this file to avoid circular
# dependencies between the files that originally contained those things.

from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from trulens.nn.backend import Tensor

# Atomic model inputs (at least from our perspective)
DataLike = Union[np.ndarray, Tensor]

# Model input arguments. Either a single element or list or tuple of several.
ArgsLike = Union[DataLike, List[DataLike], Tuple[DataLike]]

# Model input kwargs.
Kwargs = Dict[str, DataLike]

# 
FeedLike = Dict[Tensor, DataLike]

# A type to check for the latter of the above.
DATA_CONTAINER_TYPE = (list, tuple)

# Convert a single element input to the multi-input one.
def as_args(ele):
    if isinstance(ele, DATA_CONTAINER_TYPE):
        return ele
    else:
        return [ele]

# For models that take in both args and kwargs, a container:
@dataclass
class ModelInputs:
    args: List[DataLike] = field(default_factory=list)
    kwargs: Kwargs = field(default_factory=dict)

    def map(self, f):
        return ModelInputs(
            args=f(self.args),
            kwargs = {k: f(v) for k, v in self.kwargs.items()}
        )


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
InterventionLike = Union[ArgsLike, Kwargs, ModelInputs]
