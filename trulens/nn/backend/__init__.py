import os
from enum import Enum
import traceback
import importlib
from trulens.utils import tru_logger

# Do not use directly, use get_backend
_TRULENS_BACKEND_IMPL = None


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
        	self.value == Backend.TF_KERAS.value)


def get_backend(suppress_warnings=False):
    global _TRULENS_BACKEND_IMPL
    if 'TRULENS_BACKEND' in os.environ.keys():
        _TRULENS_BACKEND = Backend.from_name(os.environ['TRULENS_BACKEND'])
    else:
        _TRULENS_BACKEND = Backend.UNKNOWN
    try:
        if _TRULENS_BACKEND == Backend.PYTORCH:
            _TRULENS_BACKEND_IMPL = importlib.import_module(
                name='trulens.nn.backend.pytorch_backend.pytorch')

        elif _TRULENS_BACKEND.is_keras_derivative():
            _TRULENS_BACKEND_IMPL = importlib.import_module(
                name='trulens.nn.backend.keras_backend.keras')
            # KerasBackend has multiple backend implementations of the keras 
            # library, so reload should be called to refresh if backend changes
            # between keras vs tf.keras.
            if _TRULENS_BACKEND != _TRULENS_BACKEND_IMPL.backend:
                importlib.reload(_TRULENS_BACKEND_IMPL)

        elif _TRULENS_BACKEND == Backend.TENSORFLOW:
            _TRULENS_BACKEND_IMPL = importlib.import_module(
                name='trulens.nn.backend.tf_backend.tf')

        elif _TRULENS_BACKEND == Backend.UNKNOWN:
            if not suppress_warnings:
                tru_logger.warn(
                    'The current backend is unset or unknown. Trulens will '
                    'attempt to use any previously loaded backends, but may '
                    'cause problems. Valid backends are `pytorch`, '
                    '`tensorflow`, `keras`, and `tf.keras`. You can manually '
                    'set this with '
                    '`os.environ[\'TRULENS_BACKEND\']=backend_str`. Current '
                    'loaded backend is {}.'.format(
                        str(_TRULENS_BACKEND_IMPL)))

    except (ImportError, ModuleNotFoundError):
        _TRULENS_BACKEND_IMPL = None
        tru_logger.error(
            'Error processing backend {}. Backend is reset to None'.format(
                _TRULENS_BACKEND))
        tru_logger.error(traceback.format_exc())

    return _TRULENS_BACKEND_IMPL


_ALL_BACKEND_API_FUNCTIONS = [
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
    'sign',
    'sigmoid',
    'softmax',
    'maximum',
    'minimum',
]
