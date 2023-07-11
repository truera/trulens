import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import main
from unittest import TestCase

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tests.unit.batch_test_base import BatchTestBase
from trulens.nn.models import get_model_wrapper

assert (not tf.executing_eagerly())


class BatchTest(BatchTestBase, TestCase):

    def setUp(self):
        super(BatchTest, self).setUp()

        # Make a linear model for testing.
        x_lin = Input((self.input_size,))
        y_lin = Dense(self.output_size)(x_lin)

        self.model_lin = get_model_wrapper(Model(x_lin, y_lin))

        self.model_lin._model.set_weights(
            [self.model_lin_weights, self.model_lin_bias]
        )

        # Make a deeper model for testing.
        x_deep = Input((self.input_size,))
        y_deep = Dense(self.internal1_size)(x_deep)
        y_deep = Activation('relu')(y_deep)
        y_deep = Dense(self.internal2_size)(y_deep)
        y_deep = Activation('relu')(y_deep)
        y_deep = Dense(self.output_size)(y_deep)

        self.model_deep = get_model_wrapper(Model(x_deep, y_deep))

        self.model_deep._model.set_weights(
            [
                self.model_deep_weights_1, self.model_deep_bias_1,
                self.model_deep_weights_2, self.model_deep_bias_2,
                self.model_deep_weights_3, self.model_deep_bias_3
            ]
        )


if __name__ == '__main__':
    main()
