import os
os.environ['TRULENS_BACKEND'] = 'pytorch'

from unittest import TestCase, main

from tests.unit.doi_test_base import DoiTestBase

from torch import Tensor
from torch.nn import Linear, Module, ReLU
from trulens.nn.backend import get_backend
from trulens.nn.models import get_model_wrapper

class DoiTest(DoiTestBase, TestCase):
    def setUp(self):
        super(DoiTest, self).setUp()

        class Exp(Module):
            def __init__(this, coeff, exp):
                super(Exp, this).__init__()
                
                this.coeff = coeff
                this.exp = exp

            def forward(this, x):
                return this.coeff * (x ** this.exp)

        class DoubleExp(Module):

            def __init__(this):
                super(DoubleExp, this).__init__()

                this.layer1 = Exp(self.l1_coeff, self.l1_exp)
                this.layer2 = Exp(self.l2_coeff, self.l2_exp)
                
            def forward(this, x):
                return this.layer2(this.layer1(x))

        self.model = get_model_wrapper(
            DoubleExp(), input_shape=(1,)
        )

        self.layer0 = None
        self.layer1 = "layer1"
        self.layer2 = "layer2"


if __name__ == '__main__':
    main()
