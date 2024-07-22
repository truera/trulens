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

        self.model_lin = get_model_wrapper(
            graph_lin, input_tensors=x_lin, output_tensors=y_lin
        )

        # Make a deeper model for testing.
        graph_deep = Graph()

        with graph_deep.as_default():
            x_deep = placeholder('float32', (None, self.input_size))
            z1_deep = (
                x_deep @ self.model_deep_weights_1 + self.model_deep_bias_1
            )
            z2_deep = relu(z1_deep)
            z3_deep = (
                z2_deep @ self.model_deep_weights_2 + self.model_deep_bias_2
            )
            z4_deep = relu(z3_deep)
            y_deep = (
                z4_deep @ self.model_deep_weights_3 + self.model_deep_bias_3
            )

        self.model_deep = get_model_wrapper(
            graph_deep,
            input_tensors=x_deep,
            output_tensors=y_deep,
            internal_tensor_dict=dict(layer2=z2_deep, layer3=z3_deep)
        )

        self.layer2 = 'layer2'
        self.layer3 = 'layer3'


if __name__ == '__main__':
    main()
