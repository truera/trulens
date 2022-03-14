from unittest import main
from unittest import TestCase

from tensorflow import Graph
from tensorflow import placeholder

from tests.unit.environment_test_base import EnvironmentTestBase
from trulens.nn.backend import Backend
from trulens.nn.models.tensorflow_v1 import TensorflowModelWrapper


class EnvironmentTest(EnvironmentTestBase, TestCase):

    def setUp(self):
        super(EnvironmentTest, self).setUp()
        # Make a linear model for testing.
        graph = Graph()

        with graph.as_default():
            x_lin = placeholder('float32', (None, self.input_size))
            y_lin = x_lin @ self.model_lin_weights + self.model_lin_bias

        self.models = [graph]
        self.models_wrapper_kwargs = [
            {
                'input_tensors': x_lin,
                'output_tensors': y_lin
            }
        ]

        self.correct_backend = Backend.TENSORFLOW
        self.model_wrapper_type = TensorflowModelWrapper
