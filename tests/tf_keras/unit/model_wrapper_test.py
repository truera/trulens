import os

os.environ['TRULENS_BACKEND'] = 'tf.keras'

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from unittest import main
from unittest import TestCase

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tests.unit.model_wrapper_test_base import ModelWrapperTestBase
from trulens.nn.models.keras import KerasModelWrapper


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
            ]
        )

        self.layer0 = 0
        self.layer1 = 1
        self.layer2 = 2
        self.out = 'logits'
        """
        # kwarg handling not yet finished for keras backend
        
        X = Input((3, ), name="X")
        Coeffs = Input((3, ), name="Coeffs")
        divisor = Input((1, ), name="divisor")
        Degree = Input((3, ), name="Degree")
        offset = Input((1, ), name="offset")

        layer1 = Lambda(lambda X: X)(X)
        layer2 = Lambda(lambda ins: (ins[0] ** ins[3]) * ins[1] / ins[2] + ins[4])([layer1, Coeffs, divisor, Degree, offset])

        self.model_kwargs = KerasModelWrapper(Model([X, Coeffs, divisor, Degree, offset], layer2))
        self.model_kwargs_layer1 = 1
        self.model_kwargs_layer2 = 2
        """


if __name__ == '__main__':
    main()
