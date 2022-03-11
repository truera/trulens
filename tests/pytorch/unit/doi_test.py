import os
os.environ['TRULENS_BACKEND'] = 'pytorch'

from unittest import TestCase, main

from tests.unit.doi_test_base import DoiTestBase

from torch.nn import Module
from trulens.nn.models import get_model_wrapper


class Exponential(Module):

    def __init__(this, coeff, exp):
        super(Exponential, this).__init__()

        this.coeff = coeff
        this.exp = exp

    def forward(this, x):
        return this.coeff * (x**this.exp)


class DoubleExponential(Module):

    def __init__(this, l1_coeff, l1_exp, l2_coeff, l2_exp):
        super(DoubleExponential, this).__init__()

        this.layer1 = Exponential(l1_coeff, l1_exp)
        this.layer2 = Exponential(l2_coeff, l2_exp)

    def forward(this, x):
        return this.layer2(this.layer1(x))


class DoiTest(DoiTestBase, TestCase):

    def setUp(self):
        super(DoiTest, self).setUp()

        self.model = get_model_wrapper(
            DoubleExponential(
                self.l1_coeff, self.l1_exp, self.l2_coeff, self.l2_exp),
            input_shape=(1,))

        self.layer0 = None
        self.layer1 = "layer1"
        self.layer2 = "layer2"


if __name__ == '__main__':
    main()
