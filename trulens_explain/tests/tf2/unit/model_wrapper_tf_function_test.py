import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

import unittest
from unittest import main
from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from trulens.nn.models import get_model_wrapper

from tests.unit.model_wrapper_test_base import ModelWrapperTestBase


class TFFunctionModel(Model):

    def __init__(self):
        super(TFFunctionModel, self).__init__()
        self.dense_1 = Dense(2, activation='relu', input_shape=(2,))
        self.dense_2 = Dense(2, activation='relu')
        self.dense_3 = Dense(1, name='logits')

    @tf.function
    def call(self, x):
        z = self.dense_1(x)
        z = self.dense_2(z)
        y = self.dense_3(z)
        return y


class ModelWrapperTest(ModelWrapperTestBase, TestCase):

    def setUp(self):
        super(ModelWrapperTest, self).setUp()

        subclassed = TFFunctionModel()
        subclassed.build((5, 2))
        subclassed.set_weights(
            [
                self.layer1_weights, self.internal_bias, self.layer2_weights,
                self.internal_bias, self.layer3_weights, self.bias
            ]
        )
        self.model = get_model_wrapper(subclassed)
        self.model.set_output_layers([subclassed.dense_3])

        self.layer0 = None
        self.layer1 = 0
        self.layer2 = 1
        self.out = 'logits'

    @unittest.skip(
        'Base class uses layer 0 as multi-input but does not exist in subclass'
    )
    def test_qoibprop_multiple_inputs(self):
        return


if __name__ == '__main__':
    main()
