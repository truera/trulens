import os
os.environ['TRULENS_BACKEND'] = 'keras'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
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

    def test_wrong_keras_version(self):
        import tensorflow as tf

        x = tf.keras.layers.Input((2,))
        z = tf.keras.layers.Dense(2, activation='relu')(x)
        z = tf.keras.layers.Dense(2, activation='relu')(z)
        y = tf.keras.layers.Dense(1, name='logits')(z)

        tf_keras_model = tf.keras.models.Model(x, y)

        with self.assertRaises(ValueError):
            KerasModelWrapper(tf_keras_model)


if __name__ == '__main__':
    main()
