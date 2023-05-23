import os

os.environ['TRULENS_BACKEND'] = 'keras'

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from unittest import main
from unittest import TestCase

from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from tests.unit.attribution_axioms_test_base import AxiomsTestBase
from trulens.nn.models import get_model_wrapper


class AxiomsTest(AxiomsTestBase, TestCase):

    def setUp(self):
        super(AxiomsTest, self).setUp()

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

        self.layer2 = 2
        self.layer3 = 3


class NestedAxiomsTest(AxiomsTestBase, TestCase):

    def setUp(self):
        super(NestedAxiomsTest, self).setUp()

        # Make a linear model for testing.
        x_lin = Input((self.input_size,))
        y_lin = Dense(self.output_size)(x_lin)
        nested_model = Model(x_lin, y_lin)

        x_lin = Input((self.input_size,))
        y_lin = nested_model(x_lin)

        self.model_lin = get_model_wrapper(Model(x_lin, y_lin))

        self.model_lin._model.set_weights(
            [self.model_lin_weights, self.model_lin_bias]
        )

        # Make a deeper model for testing.
        x_deep = Input((self.input_size,))
        y_deep = Dense(self.internal1_size)(x_deep)
        y_deep = Activation('relu')(y_deep)

        # nested model definition
        nested_x_deep = Input((self.internal1_size,))
        nested_y_deep = Dense(self.internal2_size)(nested_x_deep)
        nested_y_deep = Activation('relu')(nested_y_deep)
        nested_model = Model(nested_x_deep, nested_y_deep)

        y_deep = nested_model(y_deep)
        y_deep = Dense(self.output_size)(y_deep)

        self.model_deep = get_model_wrapper(Model(x_deep, y_deep))

        self.model_deep._model.set_weights(
            [
                self.model_deep_weights_1, self.model_deep_bias_1,
                self.model_deep_weights_2, self.model_deep_bias_2,
                self.model_deep_weights_3, self.model_deep_bias_3
            ]
        )

        self.layer2 = 2
        self.layer3 = 3


if __name__ == '__main__':
    main()
