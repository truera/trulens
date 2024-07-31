import os

from trulens.nn.models import get_model_wrapper

os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import main
from unittest import TestCase

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tests.unit.doi_test_base import DoiTestBase


class DoiTest(DoiTestBase, TestCase):

    def setUp(self):
        super(DoiTest, self).setUp()

        l0 = Input((1,))
        l1 = Lambda(lambda input: self.l1_coeff * (input**self.l1_exp))(l0)
        l2 = Lambda(lambda input: self.l2_coeff * (input**self.l2_exp))(l1)

        self.model = get_model_wrapper(Model(l0, l2))

        self.layer0 = None
        self.layer1 = 1
        self.layer2 = 2


if __name__ == '__main__':
    main()
