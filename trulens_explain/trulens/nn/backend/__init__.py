from contextlib import contextmanager
from enum import Enum
import importlib
import os
import traceback
from typing import Iterable, Tuple

import numpy as np
from trulens.utils import tru_logger
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import nested_map
from trulens.utils.typing import om_of_many
from trulens.utils.typing import TensorAKs
from trulens.utils.typing import Tensors

# Do not use directly, use get_backend
_TRULENS_BACKEND_IMPL = None

# Each backend module provides:
# Tensor
# dim_order
# channel_axis
# backend
# floatX
# floatX_size


class OutOfMemory(RuntimeError):

    def __init__(self, settings, **kwargs):
        self.kwargs = kwargs
        self.settings = settings

    def __str__(self):
        message = f'Ran out of memory ({self.kwargs}). Consider reducing memory-impactful parameters.\n'
        for setting in self.settings:
            message += '\t' + setting + '\n'

        return message


@contextmanager
def memory_suggestions(*settings, call_before=None, call_after=None, **kwargs):
    """
    Context manager to catch device memory issues and report better suggestions
    than default exceptions.

    Usage:

        with memory_suggestions("batch size=1000"):
            # Something memory intensive that could be helped by reducing batch
            # size.

    This will catch out of memory issues and report the messages included in the
    memory_suggestions calls as suggestions for parameters to adjust. You can
    stack these and put suggestions indicative of memory-related settings like
    batch sizes, intervention sizes, data types, etc. Inside the context,
    another memory_suggestions can be used with more messages which will be
    combined for reporting if memory does run out.

    Before/after call are for extending this method with additional tasks like
    default device handling in pytorch.
    """

    settings = om_of_many(list(settings))
    state = None

    try:
        if call_before is not None:
            state = call_before()

        yield state

    except OutOfMemory as e:
        raise OutOfMemory(settings=e.settings + settings, **e.kwargs)

    except RuntimeError as e:
        if 'out of memory' in str(e):  # cuda out of memory exception
            raise OutOfMemory(settings=settings, **kwargs)
        else:
            # TODO: catch similar exceptions in other backends
            raise

    finally:
        if call_after is not None:
            call_after(state)


def rebatch(vals: Tensors,
            *extra_vals: Tuple[Tensors, ...],
            batch_size=None) -> Iterable[Tuple[Tensors, ...]]:
    """Rebatch the values in `vals` into bins of size `batch_size`. If more sets
    of values are given in `extra_vals`, those are batched into the same bins as
    well."""
    B = get_backend()
    original_batch_size = vals.first_batchable(B).shape[0]

    if batch_size is None:
        batch_size = original_batch_size

    def take(batch_idx):

        def f(val):
            if val.shape[0] != original_batch_size:
                return val
            else:
                return val[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        return f

    all_vals: Tuple[ModelInputs, ...] = (vals,) + extra_vals

    for batch_idx in range(0, 1 + original_batch_size // batch_size):
        if batch_idx * batch_size >= original_batch_size:
            continue

        yield tuple(map(lambda v: v.map(take(batch_idx)), all_vals))


def tile(what: TensorAKs, onto: TensorAKs) -> TensorAKs:
    """Tiles elements of `what` some number of times so they have the same first
    dimension size as `onto`. Picks the number of tiles from the first of each
    container and skips tiling anything in `what` that does not have the same
    first dimension as the first item in that container."""
    B = get_backend()

    inputs_dim = what.first_batchable(B).shape[0]
    expected_dim = onto.first_batchable(B).shape[0]
    doi_resolution = int(expected_dim // inputs_dim)

    def tile_val(val):
        """Tile the given value if expected_dim matches val's first
        dimension. Otherwise return original val unchanged."""

        if val.shape[0] != inputs_dim:
            tru_logger.warning(
                f'Value {val} of shape {val.shape} is assumed to not be '
                f'batchable due to its shape not matching prior batchable '
                f'values of shape ({inputs_dim},...). If this is '
                f'incorrect, make sure its first dimension matches prior '
                f'batchable values.'
            )
            return val

        tile_shape = [1 for _ in range(len(val.shape))]
        tile_shape[0] = doi_resolution
        repeat_shape = tuple(tile_shape)

        if isinstance(val, np.ndarray):
            return np.tile(val, repeat_shape)
        elif B.is_tensor(val):
            return B.tile(val, repeat_shape)
        else:
            tru_logger.debug(
                f'Ignoring tiling of unhandled val {val.__class__.__name__}'
            )
            return val

    return what.map(tile_val)


class Backend(Enum):
    PYTORCH = 1
    TENSORFLOW = 2
    KERAS = 3
    TF_KERAS = 4
    UNKNOWN = 5

    @classmethod
    def from_name(cls, name):
        name = name.lower()
        if name == 'pytorch' or name == 'torch':
            return Backend.PYTORCH
        elif name == 'tensorflow' or name == 'tf':
            return Backend.TENSORFLOW
        elif name == 'keras':
            return Backend.KERAS
        elif name == 'tf.keras' or name == 'tf_keras':
            return Backend.TF_KERAS
        else:
            return Backend.UNKNOWN

    def is_keras_derivative(self):
        return (
            self.value == Backend.KERAS.value or
            self.value == Backend.TF_KERAS.value
        )


def get_backend(suppress_warnings=False):
    global _TRULENS_BACKEND_IMPL
    if 'TRULENS_BACKEND' in os.environ.keys():
        _TRULENS_BACKEND = Backend.from_name(os.environ['TRULENS_BACKEND'])
    else:
        _TRULENS_BACKEND = Backend.UNKNOWN
    try:
        if _TRULENS_BACKEND == Backend.PYTORCH:
            _TRULENS_BACKEND_IMPL = importlib.import_module(
                name='trulens.nn.backend.pytorch_backend.pytorch'
            )

        elif _TRULENS_BACKEND.is_keras_derivative():
            _TRULENS_BACKEND_IMPL = importlib.import_module(
                name='trulens.nn.backend.keras_backend.keras'
            )
            # KerasBackend has multiple backend implementations of the keras
            # library, so reload should be called to refresh if backend changes
            # between keras vs tf.keras.
            if _TRULENS_BACKEND != _TRULENS_BACKEND_IMPL.backend:
                importlib.reload(_TRULENS_BACKEND_IMPL)

        elif _TRULENS_BACKEND == Backend.TENSORFLOW:
            _TRULENS_BACKEND_IMPL = importlib.import_module(
                name='trulens.nn.backend.tf_backend.tf'
            )

        elif _TRULENS_BACKEND == Backend.UNKNOWN:
            if not suppress_warnings:
                tru_logger.warning(
                    'The current backend is unset or unknown. Trulens will '
                    'attempt to use any previously loaded backends, but may '
                    'cause problems. Valid backends are `pytorch`, '
                    '`tensorflow`, `keras`, and `tf.keras`. You can manually '
                    'set this with '
                    '`os.environ[\'TRULENS_BACKEND\']=backend_str`. Current '
                    'loaded backend is {}.'.format(str(_TRULENS_BACKEND_IMPL))
                )

    except (ImportError, ModuleNotFoundError):
        _TRULENS_BACKEND_IMPL = None
        tru_logger.error(
            'Error processing backend {}. Backend is reset to None'.
            format(_TRULENS_BACKEND)
        )
        tru_logger.error(traceback.format_exc())

    return _TRULENS_BACKEND_IMPL


_ALL_BACKEND_API_FUNCTIONS = [
    'set_seed',
    'is_deterministic',
    'dim_order',
    'channel_axis',
    'floatX',
    'backend',
    'gradient',
    'as_array',
    'as_tensor',
    'is_tensor',
    'int_shape',
    'shape',
    'expand_dims',
    'reshape',
    'mean',
    'sum',
    'abs',
    'max',
    'ones_like',
    'zeros_like',
    'random_normal_like',
    'clone',
    'stack',
    'tile',
    'concat',
    'sign',
    'sigmoid',
    'softmax',
    'maximum',
    'minimum',
]
