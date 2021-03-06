import os

from unittest import TestCase, main
from torch.nn import Linear, Module

from tests.unit.environment_test_base import EnvironmentTestBase
from trulens.nn.models.pytorch import PytorchModelWrapper
from trulens.nn.backend import get_backend


class EnvironmentTest(EnvironmentTestBase, TestCase):

    def setUp(self):
        super(EnvironmentTest, self).setUp()

        # Make a linear model for testing.
        class M(Module):

            def __init__(this):
                super(M, this).__init__()
                this.layer = Linear(self.input_size, self.output_size)
                B = get_backend()
                this.layer.weight.data = B.as_tensor(self.model_lin_weights.T)
                this.layer.bias.data = B.as_tensor(self.model_lin_bias)

            def forward(this, x):
                return this.layer(x)

        self.models = [M()]
        self.models_wrapper_kwargs = [{'input_shape': (self.input_size,)}]
        self.correct_backend = 'pytorch'
        self.model_wrapper_type = PytorchModelWrapper
