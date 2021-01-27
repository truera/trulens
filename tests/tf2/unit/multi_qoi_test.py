import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import TestCase, main

import numpy as np

from trulens.nn import backend as B
from trulens.nn.models import ModelWrapper

import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model

from tests.unit.multi_qoi_test_base import MultiQoiTestBase


class MultiQoiTest(MultiQoiTestBase, TestCase):

    def test_per_timestep(self):
        num_classes = 5
        num_features = 3
        num_timesteps = 4
        num_hidden_state = 10
        batch_size = 32

        base_model = Sequential(
            [
                Input(shape=(num_timesteps, num_features)),
                GRU(num_hidden_state, name="rnn", return_sequences=True),
                Dense(num_classes, name="dense"),
            ])

        model = ModelWrapper(base_model)
        super(MultiQoiTest, self).per_timestep_qoi(
            model, num_classes, num_features, num_timesteps, batch_size)


if __name__ == '__main__':
    main()
