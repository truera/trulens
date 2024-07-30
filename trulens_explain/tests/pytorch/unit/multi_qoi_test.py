import os

os.environ["TRULENS_BACKEND"] = "pytorch"

from unittest import TestCase
from unittest import main

import torch
from torch.nn import GRU
from torch.nn import Linear
from torch.nn import Module
from trulens.nn.models import get_model_wrapper

from tests.unit.multi_qoi_test_base import MultiQoiTestBase


class MultiQoiTest(MultiQoiTestBase, TestCase):
    def test_per_timestep(self):
        num_classes = 5
        num_features = 3
        num_timesteps = 4
        num_hidden_state = 10
        batch_size = 32

        class M(Module):
            def __init__(self):
                super().__init__()
                self.rnn = GRU(num_features, num_hidden_state)
                self.dense = Linear(num_hidden_state, num_classes)

            def forward(self, x):
                z1 = self.rnn(x)
                z2 = self.dense(z1[0])
                return z2

        torch.backends.cudnn.enabled = False
        # We have the same problem as this: https://github.com/pytorch/captum/issues/564

        model = get_model_wrapper(M())
        super().per_timestep_qoi(
            model, num_classes, num_features, num_timesteps, batch_size
        )

        torch.backends.cudnn.enabled = True


if __name__ == "__main__":
    main()
