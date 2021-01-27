import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import TestCase, main

import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.models import Model

from trulens.nn.models import ModelWrapper
from tests.unit.batch_test_base import BatchTestBase

assert (not tf.executing_eagerly())


class BatchTest(BatchTestBase, TestCase):

    def setUp(self):
        super(BatchTest, self).setUp()

        # Make a linear model for testing.
        x_lin = Input((self.input_size,))
        y_lin = Dense(self.output_size)(x_lin)

        self.model_lin = ModelWrapper(Model(x_lin, y_lin))

        self.model_lin._model.set_weights(
            [self.model_lin_weights, self.model_lin_bias])

        # Make a deeper model for testing.
        x_deep = Input((self.input_size,))
        y_deep = Dense(self.internal1_size)(x_deep)
        y_deep = Activation('relu')(y_deep)
        y_deep = Dense(self.internal2_size)(y_deep)
        y_deep = Activation('relu')(y_deep)
        y_deep = Dense(self.output_size)(y_deep)

        self.model_deep = ModelWrapper(Model(x_deep, y_deep))

        self.model_deep._model.set_weights(
            [
                self.model_deep_weights_1, self.model_deep_bias_1,
                self.model_deep_weights_2, self.model_deep_bias_2,
                self.model_deep_weights_3, self.model_deep_bias_3
            ])


if __name__ == '__main__':
    main()
