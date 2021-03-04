import os
os.environ['TRULENS_BACKEND'] = 'pytorch'

import numpy as np

from torch import Tensor
from torch.nn import Linear, Module, ReLU
from unittest import TestCase, main

from trulens.nn.backend import get_backend
from trulens.nn.models.pytorch import PytorchModelWrapper
from trulens.nn.quantities import MaxClassQoI
from trulens.nn.slices import Cut
from tests.unit.model_wrapper_test_base import ModelWrapperTestBase


class ModelWrapperTest(ModelWrapperTestBase, TestCase):

    def setUp(self):
        super(ModelWrapperTest, self).setUp()

        class M(Module):

            def __init__(this):
                super(M, this).__init__()
                this.l1 = Linear(2, 2)
                this.l1_relu = ReLU()
                this.l2 = Linear(2, 2)
                this.l2_relu = ReLU()
                this.logits = Linear(2, 1)

                B = get_backend()
                this.l1.weight.data = B.as_tensor(self.layer1_weights.T)
                this.l1.bias.data = B.as_tensor(self.internal_bias)
                this.l2.weight.data = B.as_tensor(self.layer2_weights.T)
                this.l2.bias.data = B.as_tensor(self.internal_bias)
                this.logits.weight.data = B.as_tensor(self.layer3_weights.T)
                this.logits.bias.data = B.as_tensor(self.bias)

            def forward(this, x):
                x = this.l1(x)
                x = this.l1_relu(x)
                x = this.l2(x)
                x = this.l2_relu(x)
                return this.logits(x)

        self.model = PytorchModelWrapper(M(), (2,))

        self.layer0 = None
        self.layer1 = 'l1_relu'
        self.layer2 = 'l2_relu'

    # Overriden tests.

    def test_qoibprop_multiple_inputs(self):
        r = self.model.qoi_bprop(
            MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
            attribution_cut=Cut(['l1', 'l2'], anchor='in'))

        self.assertEqual(len(r), 2)
        self.assertTrue(np.allclose(r[0], np.array([[3., -1.], [0., 2.]])))
        self.assertTrue(np.allclose(r[1], np.array([[3., 2.], [2., 2.]])))


if __name__ == '__main__':
    main()
