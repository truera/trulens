from netlens.backend.load_backend import _BACKEND

if _BACKEND == 'pytorch':
    from netlens.backend.pytorch_backend.pytorch import *
elif _BACKEND == 'keras' or _BACKEND == 'tf.keras':
    from netlens.backend.keras_backend.keras import *
elif _BACKEND == 'tensorflow' or _BACKEND == 'tf':
    from netlens.backend.tf_backend.tf import *
