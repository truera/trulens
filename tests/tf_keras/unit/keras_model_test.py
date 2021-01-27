import os
os.environ['TRULENS_BACKEND'] = 'tf.keras'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from unittest import TestCase, main

from trulens.nn.models.keras import KerasModelWrapper
from tests.unit.model_wrapper_test_base import ModelWrapperTestBase


class ModelWrapperTest(ModelWrapperTestBase, TestCase):

    def setUp(self):
        super(ModelWrapperTest, self).setUp()

        x = Input((2,))
        z = Dense(2, activation='relu')(x)
        z = Dense(2, activation='relu')(z)
        y = Dense(1, name='logits')(z)

        self.model = KerasModelWrapper(Model(x, y))

        self.model._model.set_weights(
            [
                self.layer1_weights, self.internal_bias, self.layer2_weights,
                self.internal_bias, self.layer3_weights, self.bias
            ])

        self.layer0 = 0
        self.layer1 = 1
        self.layer2 = 2


if __name__ == '__main__':
    main()
