import os
os.environ['NETLENS_BACKEND'] = 'pytorch'

from unittest import TestCase, main

import numpy as np

from netlens import backend as B
from netlens.models import ModelWrapper

from torch import cat
from torch.nn import GRU, Linear, Module

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
                super(M, self).__init__()
                self.rnn = GRU(num_features, num_hidden_state)
                self.dense = Linear(num_hidden_state, num_classes)
                
            def forward(self, x):
                z1 = self.rnn(x)
                z2 = self.dense(z1[0])
                return z2

        model = ModelWrapper(M(), (num_timesteps, num_features))
        super(MultiQoiTest, self).per_timestep_qoi(
            model, 
            num_classes,
            num_features, 
            num_timesteps,
            batch_size
        )

if __name__ == '__main__':
    main()
