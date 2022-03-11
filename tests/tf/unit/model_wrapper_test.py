import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from tensorflow import Graph, placeholder
from tensorflow.nn import relu
from unittest import TestCase, main

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
            graph, x, y, dict(x=x, z1=z1, z2=z2, logits=y))

        self.layer0 = 'x'
        self.layer1 = 'z1'
        self.layer2 = z2.name
        self.out = y.name


if __name__ == '__main__':
    main()
