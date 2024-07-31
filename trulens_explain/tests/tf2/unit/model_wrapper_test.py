import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import main
from unittest import TestCase

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tests.unit.model_wrapper_test_base import ModelWrapperTestBase
from trulens.nn.models import get_model_wrapper


class ModelWrapperTest(ModelWrapperTestBase, TestCase):

    def setUp(self):
        super(ModelWrapperTest, self).setUp()

        x = Input((2,))
        z = Dense(2, activation='relu')(x)
        z = Dense(2, activation='relu')(z)
        y = Dense(1, name='logits')(z)

        self.model = get_model_wrapper(Model(x, y))

        self.model._model.set_weights(
            [
                self.layer1_weights, self.internal_bias, self.layer2_weights,
                self.internal_bias, self.layer3_weights, self.bias
            ]
        )

        self.layer0 = 0
        self.layer1 = 1
        self.layer2 = 2
        self.out = 'logits'


if __name__ == '__main__':
    main()
