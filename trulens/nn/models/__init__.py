
from trulens.nn import backend as B

if B.backend == 'keras' or B.backend == 'tf.keras':
    from trulens.nn.models.keras import KerasModelWrapper as ModelWrapper

elif B.backend == 'pytorch':
    from trulens.nn.models.pytorch import PytorchModelWrapper as ModelWrapper

elif B.backend == 'tensorflow':
    import tensorflow as tf

    if tf.__version__.startswith('2'):
        from trulens.nn.models.tensorflow_v2 import Tensorflow2ModelWrapper as ModelWrapper
    else:
        from trulens.nn.models.tensorflow_v1 import TensorflowModelWrapper as ModelWrapper

from trulens.nn.models._model_base import ModelWrapper as AbstractModelWrapper
