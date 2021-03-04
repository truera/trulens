import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from tensorflow import Graph, placeholder
from tensorflow.nn import relu
from unittest import TestCase, main

from trulens.nn.models import get_model_wrapper
from tests.unit.attribution_axioms_test_base import AxiomsTestBase


class AxiomsTest(AxiomsTestBase, TestCase):

    def setUp(self):
        super(AxiomsTest, self).setUp()

        # Make a linear model for testing.
        graph_lin = Graph()

        with graph_lin.as_default():
            x_lin = placeholder('float32', (None, self.input_size))
            y_lin = x_lin @ self.model_lin_weights + self.model_lin_bias

        self.model_lin = ModelWrapper(graph_lin, x_lin, y_lin)

        # Make a deeper model for testing.
        graph_deep = Graph()

        with graph_deep.as_default():
            x_deep = placeholder('float32', (None, self.input_size))
            z1_deep = (
                x_deep @ self.model_deep_weights_1 + self.model_deep_bias_1)
            z2_deep = relu(z1_deep)
            z3_deep = (
                z2_deep @ self.model_deep_weights_2 + self.model_deep_bias_2)
            z4_deep = relu(z3_deep)
            y_deep = (
                z4_deep @ self.model_deep_weights_3 + self.model_deep_bias_3)

        self.model_deep = ModelWrapper(
            graph_deep, x_deep, y_deep, dict(layer2=z2_deep, layer3=z3_deep))

        self.layer2 = 'layer2'
        self.layer3 = 'layer3'


if __name__ == '__main__':
    main()
