"""Type aliases and data container classes. """

# NOTE(piotrm) had to move some things to this file to avoid circular
# dependencies between the files that originally contained those things.

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, Any, List[Union[np.ndarray, Any]]]

DATA_CONTAINER_TYPE = (list, tuple)


@dataclass
class ModelInputs:
    args: List[ArrayLike] = field(default_factory=list)
    kwargs: Dict[str, ArrayLike] = field(default_factory=dict)


# Baselines are either explicit or computable from the same data as sent to DoI
# __call__ .
BaselineLike = Union[ArrayLike, Callable[[ArrayLike, Optional[ModelInputs]],
                                         ArrayLike]]
