import os

os.environ["TRULENS_BACKEND"] = "tensorflow"

from unittest import TestCase
from unittest import main

import tensorflow as tf

from tests.unit.doi_test_base import DoiTestBase

assert not tf.executing_eagerly()

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from trulens.nn.models.keras import KerasModelWrapper


class DoiTest(DoiTestBase, TestCase):
    def setUp(self):
        super().setUp()

        l0 = Input((1,))
        l1 = Lambda(lambda input: self.l1_coeff * (input**self.l1_exp))(l0)
        l2 = Lambda(lambda input: self.l2_coeff * (input**self.l2_exp))(l1)

        self.model = KerasModelWrapper(Model(l0, l2))

        self.layer0 = 0
        self.layer1 = 1
        self.layer2 = 2


if __name__ == "__main__":
    main()