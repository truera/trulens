import os
from unittest import TestCase

from torch.nn import Linear
from torch.nn import Module
from trulens.nn.backend import Backend
from trulens.nn.backend import get_backend
from trulens.nn.models.pytorch import PytorchModelWrapper

from tests.unit.environment_test_base import EnvironmentTestBase


class EnvironmentTest(EnvironmentTestBase, TestCase):
    def setUp(self):
        super().setUp()

        # Make a linear model for testing.
        class M(Module):
            def __init__(this):
                super().__init__()
                this.layer = Linear(self.input_size, self.output_size)
                os.environ["TRULENS_BACKEND"] = "pytorch"
                B = get_backend()
                this.layer.weight.data = B.as_tensor(self.model_lin_weights.T)
                this.layer.bias.data = B.as_tensor(self.model_lin_bias)

            def forward(this, x):
                return this.layer(x)

        self.models = [M()]
        self.models_wrapper_kwargs = [{}]
        self.correct_backend = Backend.PYTORCH
        self.model_wrapper_type = PytorchModelWrapper
