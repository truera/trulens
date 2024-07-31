from unittest import TestCase

import numpy as np
from trulens.nn.backend import get_backend


class BackendTestBase(TestCase):

    def setUp(self):
        B = get_backend()

        self.zeros = np.zeros((3, 3))
        self.ones = np.ones((3, 3))
        self.twos = 2 * np.ones((3, 3))

        self.tzeros = B.as_tensor(self.zeros)
        self.tones = B.as_tensor(self.ones)
        self.ttwos = B.as_tensor(self.twos)

    def test_concat(self):
        B = get_backend()

        catted = np.concatenate([self.zeros, self.ones, self.twos])
        tcatted = B.concat([self.tzeros, self.tones, self.ttwos])

        self.assertTrue(np.allclose(catted, B.as_array(tcatted)))

        catted = np.concatenate([self.zeros, self.ones, self.twos])
        tcatted = B.concat(
            tuple(reversed([self.tzeros, self.tones, self.ttwos]))
        )

        self.assertFalse(np.allclose(catted, B.as_array(tcatted)))

    def test_concat_axis(self):
        B = get_backend()

        catted = np.concatenate([self.zeros, self.ones, self.twos], axis=1)
        tcatted = B.concat([self.tzeros, self.tones, self.ttwos], axis=1)

        self.assertTrue(np.allclose(catted, B.as_array(tcatted)))

        catted = np.concatenate([self.zeros, self.ones, self.twos], axis=0)
        tcatted = B.concat([self.tzeros, self.tones, self.ttwos], axis=1)

        with self.assertRaises(ValueError):
            np.allclose(catted, B.as_array(tcatted))

    def test_concat_last(self):
        B = get_backend()

        catted = np.concatenate([self.zeros, self.ones, self.twos], axis=-1)
        tcatted = B.concat([self.tzeros, self.tones, self.ttwos], axis=-1)

        self.assertTrue(np.allclose(catted, B.as_array(tcatted)))
