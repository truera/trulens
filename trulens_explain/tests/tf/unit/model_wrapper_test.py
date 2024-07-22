import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import importlib
from unittest import main
from unittest import TestCase

import tensorflow as tf

if tf.__version__.startswith('1'):
    from tensorflow import Graph
    from tensorflow import placeholder
    from tensorflow.nn import relu
else:
    raise ImportError(
        f'Running Tensorflow 1 tests with incorrect version of Tensorflow. Expected 1.x, got {tf.__version__}'
    )

from trulens.nn.models.tensorflow_v1 import TensorflowModelWrapper

from tests.unit.model_wrapper_test_base import ModelWrapperTestBase


class ModelWrapperTest(ModelWrapperTestBase, TestCase):

    def setUp(self):
        super(ModelWrapperTest, self).setUp()

        graph = Graph()
        with graph.as_default():
            x = placeholder('float32', (None, 2))
            z1 = relu(x @ self.layer1_weights + self.internal_bias)
            z2 = relu(z1 @ self.layer2_weights + self.internal_bias)
            y = z2 @ self.layer3_weights + self.bias

        self.model = TensorflowModelWrapper(
            graph,
            input_tensors=x,
            output_tensors=y,
            internal_tensor_dict=dict(x=x, z1=z1, z2=z2, logits=y)
        )

        self.layer0 = 'x'
        self.layer1 = 'z1'
        self.layer2 = z2.name
        self.out = y.name

        # kwarg handling not yet implemented for tf backend:
        """
        graph = Graph()
        with graph.as_default():
            # args
            X = placeholder('float32', (None, 3), name="X")
            Coeffs = placeholder('float32', (None, 3), name="Coeffs")
            divisor = placeholder('float32', (None, 1), name="divisor")

            # kwargs for backends that support them
            Degree = placeholder('float32', (None, 3), name="Degree")
            offset = placeholder('float32', (None, 1), name="offset")

            layer1 = X
            layer2 = (layer1**Degree) * Coeffs / divisor + offset
            y = layer2

        # args must be provided first in input tensors, otherwise tf1 wrapper kwargs support fails
        self.model_kwargs = TensorflowModelWrapper(
            graph, (X, Coeffs, divisor, Degree, offset), y,
            dict(
                X=X,
                Coeffs=Coeffs,
                Degree=Degree,
                divisor=divisor,
                offset=offset,
                layer1=layer1,
                layer2=layer2))

        self.model_kwargs_layer1 = "layer1"
        self.model_kwargs_layer2 = "layer2"
        """


if __name__ == '__main__':
    main()
