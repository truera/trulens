import os

os.environ['TRULENS_BACKEND'] = 'keras'

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from unittest import main
from unittest import TestCase

from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
import numpy as np
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

    def test_wrong_keras_version(self):
        import tensorflow as tf

        x = tf.keras.layers.Input((2,))
        z = tf.keras.layers.Dense(2, activation='relu')(x)
        z = tf.keras.layers.Dense(2, activation='relu')(z)
        y = tf.keras.layers.Dense(1, name='logits')(z)

        tf_keras_model = tf.keras.models.Model(x, y)

        with self.assertRaises(ValueError):
            KerasModelWrapper(tf_keras_model)


class NestedModelWrapperTest(ModelWrapperTestBase, TestCase):

    def setUp(self):
        super(NestedModelWrapperTest, self).setUp()
        n_x = Input((2,))
        n_y = Dense(2, activation='relu')(n_x)
        nested_model = Model([n_x], [n_y])

        x = Input((2,))
        z = nested_model(x)
        z = Dense(2, activation='relu')(z)
        y = Dense(1, name='logits')(z)
        model = Model(x, y)

        self.model = KerasModelWrapper(model)

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
