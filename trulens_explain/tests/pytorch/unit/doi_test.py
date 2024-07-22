import os

os.environ['TRULENS_BACKEND'] = 'pytorch'

from unittest import main
from unittest import TestCase

from torch import Tensor
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from trulens.nn.backend import get_backend
from trulens.nn.models import get_model_wrapper

from tests.unit.doi_test_base import DoiTestBase


class DoiTest(DoiTestBase, TestCase):

    def setUp(self):
        super(DoiTest, self).setUp()

        class Exponential(Module):

            def __init__(this, coeff, exp):
                super(Exponential, this).__init__()

                this.coeff = coeff
                this.exp = exp

            def forward(this, x):
                return this.coeff * (x**this.exp)

        class DoubleExponential(Module):

            def __init__(this):
                super(DoubleExponential, this).__init__()

                this.layer1 = Exponential(self.l1_coeff, self.l1_exp)
                this.layer2 = Exponential(self.l2_coeff, self.l2_exp)

            def forward(this, x):
                return this.layer2(this.layer1(x))

        self.model = get_model_wrapper(DoubleExponential())

        self.layer0 = None
        self.layer1 = 'layer1'
        self.layer2 = 'layer2'


if __name__ == '__main__':
    main()
