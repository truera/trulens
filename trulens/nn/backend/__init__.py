import os
import importlib

# Do not use directly, use get_backend
_TRULENS_BACKEND_IMPL = None
def get_backend():
    global _TRULENS_BACKEND_IMPL
    if 'TRULENS_BACKEND' in os.environ.keys():
        _TRULENS_BACKEND = os.environ['TRULENS_BACKEND']
    else:
        _TRULENS_BACKEND = 'pytorch'
    

    if _TRULENS_BACKEND == 'pytorch':
        _TRULENS_BACKEND_IMPL = importlib.import_module(name='trulens.nn.backend.pytorch_backend.pytorch')
    elif _TRULENS_BACKEND == 'keras' or _TRULENS_BACKEND == 'tf.keras':
        
        _TRULENS_BACKEND_IMPL = importlib.import_module(name='trulens.nn.backend.keras_backend.keras')
        # KerasBackend has multiple backend implementations of the keras library, 
        # so reload should be called to refresh if backend changes between keras vs tf.keras
        if _TRULENS_BACKEND != _TRULENS_BACKEND_IMPL.backend:
            importlib.reload(_TRULENS_BACKEND_IMPL)
    elif _TRULENS_BACKEND == 'tensorflow' or _TRULENS_BACKEND == 'tf':
        _TRULENS_BACKEND_IMPL = importlib.import_module(name='trulens.nn.backend.tf_backend.tf')
    return _TRULENS_BACKEND_IMPL
    
_ALL_BACKEND_API_FUNCTIONS = [
    'dim_order',
    'channel_axis',
    'Tensor',
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
