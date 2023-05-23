from unittest import main
from unittest import TestCase

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tests.unit.environment_test_base import EnvironmentTestBase
from trulens.nn.backend import Backend
from trulens.nn.models.tensorflow_v2 import Tensorflow2ModelWrapper


class TFSubclassModel(Model):

    def __init__(self):
        super(TFSubclassModel, self).__init__()
        self.dense_1 = Dense(2, activation='relu', input_shape=(2,))
        self.dense_2 = Dense(1, name='logits')

    def call(self, x):
        z = self.dense_1(x)
        y = self.dense_2(z)
        return y


class TFFunctionModel(Model):

    def __init__(self):
        super(TFFunctionModel, self).__init__()
        self.dense_1 = Dense(2, activation='relu', input_shape=(2,))
        self.dense_2 = Dense(1, name='logits')

    @tf.function
    def call(self, x):
        z = self.dense_1(x)
        y = self.dense_2(z)
        return y


class EnvironmentTest(EnvironmentTestBase, TestCase):

    def setUp(self):
        super(EnvironmentTest, self).setUp()
        # Make a linear model for testing.
        x = Input((2,))
        z = Dense(2, activation='relu')(x)
        y = Dense(1, name='logits')(z)

        self.models = [Model(x, y), TFSubclassModel(), TFFunctionModel()]
        self.models_wrapper_kwargs = [{}, {}, {}]
        self.correct_backend = Backend.TENSORFLOW
        self.model_wrapper_type = Tensorflow2ModelWrapper
