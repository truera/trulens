import os

os.environ['TRULENS_BACKEND'] = 'pytorch'

from unittest import main
from unittest import TestCase

import numpy as np
from torch import Tensor
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
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

        self.model = PytorchModelWrapper(M())

        self.layer0 = None
        self.layer1 = 'l1_relu'
        self.layer2 = 'l2_relu'
        self.out = 'logits'

        class MkwargsIdent(Module):
            """Does nothing but lets us build some inner layers."""

            def forward(this, X):
                return X

        class Mkwargs(Module):

            def __init__(self):
                super(Mkwargs, self).__init__()
                self.layer1 = MkwargsIdent()
                self.layer2 = MkwargsIdent()

            def forward(this, X, Coeffs, divisor, **kwargs):
                # Hoping kwargs forces the remaining arguments to be passed only by kwargs.

                # Capital vars are batched, lower-case ones are not.
                Degree = kwargs['Degree']
                offset = kwargs['offset']

                layer1 = this.layer1(X)

                # Capital-named vars should be batched, others should not be.
                layer2 = this.layer2(
                    (layer1**Degree) * Coeffs / divisor + offset
                )

                return layer2

        self.model_kwargs = PytorchModelWrapper(Mkwargs())
        self.model_kwargs_layer1 = 'layer1'
        self.model_kwargs_layer2 = 'layer2'

    # Overriden tests.

    def test_qoibprop_multiple_inputs(self):
        r = self.model.qoi_bprop(
            MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
            attribution_cut=Cut(['l1', 'l2'], anchor='in')
        )

        self.assertEqual(len(r), 2)
        self.assertTrue(np.allclose(r[0], np.array([[3., -1.], [0., 2.]])))
        self.assertTrue(np.allclose(r[1], np.array([[3., 2.], [2., 2.]])))


if __name__ == '__main__':
    main()
