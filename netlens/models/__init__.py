#from __future__ import absolute_import

from netlens import backend as B

if B.backend == 'keras' or B.backend == 'tf.keras':
    from netlens.models.keras import KerasModelWrapper as ModelWrapper

elif B.backend == 'pytorch':
    from netlens.models.pytorch import PytorchModelWrapper as ModelWrapper

elif B.backend == 'tensorflow':
    import tensorflow as tf

    if tf.__version__.startswith('2'):
        from netlens.models.tensorflow_v2 import Tensorflow2ModelWrapper as ModelWrapper
    else:
        from netlens.models.tensorflow_v1 import TensorflowModelWrapper as ModelWrapper

from netlens.models._model_base import ModelWrapper as AbstractModelWrapper
