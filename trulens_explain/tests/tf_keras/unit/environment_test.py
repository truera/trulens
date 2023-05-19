from unittest import main
from unittest import TestCase

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tests.unit.environment_test_base import EnvironmentTestBase
from trulens.nn.backend import Backend
from trulens.nn.models.keras import KerasModelWrapper


class EnvironmentTest(EnvironmentTestBase, TestCase):

    def setUp(self):
        super(EnvironmentTest, self).setUp()
        # Make a linear model for testing.
        x_lin = Input((self.input_size,))
        y_lin = Dense(self.output_size)(x_lin)

        self.models = [Model(x_lin, y_lin)]
        self.models_wrapper_kwargs = [{}]
        self.correct_backend = Backend.TF_KERAS
        self.model_wrapper_type = KerasModelWrapper
