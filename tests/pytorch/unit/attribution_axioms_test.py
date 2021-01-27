import os
os.environ['TRULENS_BACKEND'] = 'pytorch'

from torch import Tensor
from torch.nn import Linear, Module, ReLU
from unittest import TestCase, main

from trulens.nn import backend as B
from trulens.nn.models import ModelWrapper
from tests.unit.attribution_axioms_test_base import AxiomsTestBase


class AxiomsTest(AxiomsTestBase, TestCase):

    def setUp(self):
        super(AxiomsTest, self).setUp()

        # Make a linear model for testing.
        class M_lin(Module):

            def __init__(this):
                super(M_lin, this).__init__()
                this.layer = Linear(self.input_size, self.output_size)

                this.layer.weight.data = B.as_tensor(self.model_lin_weights.T)
                this.layer.bias.data = B.as_tensor(self.model_lin_bias)

            def forward(this, x):
                return this.layer(x)

        self.model_lin = ModelWrapper(M_lin(), (self.input_size,))

        # Make a deeper model for testing.
        class M_deep(Module):

            def __init__(this):
                super(M_deep, this).__init__()
                this.l1 = Linear(self.input_size, self.internal1_size)
                this.l1_relu = ReLU()
                this.l2 = Linear(self.internal1_size, self.internal2_size)
                this.l2_relu = ReLU()
                this.l3 = Linear(self.internal2_size, self.output_size)

                this.l1.weight.data = B.as_tensor(self.model_deep_weights_1.T)
                this.l1.bias.data = B.as_tensor(self.model_deep_bias_1)
                this.l2.weight.data = B.as_tensor(self.model_deep_weights_2.T)
                this.l2.bias.data = B.as_tensor(self.model_deep_bias_2)
                this.l3.weight.data = B.as_tensor(self.model_deep_weights_3.T)
                this.l3.bias.data = B.as_tensor(self.model_deep_bias_3)

            def forward(this, x):
                x = this.l1(x)
                x = this.l1_relu(x)
                x = this.l2(x)
                x = this.l2_relu(x)
                return this.l3(x)

        self.model_deep = ModelWrapper(M_deep(), (self.input_size,))

        self.layer2 = 'l1_relu'
        self.layer3 = 'l2'


if __name__ == '__main__':
    main()
