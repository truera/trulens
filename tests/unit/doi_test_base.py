import numpy as np

from trulens.nn import backend as B
from trulens.nn.distributions import PointDoi, LinearDoi, GaussianDoi


class DoiTestBase(object):

    def setUp(self):
        # Create an example tensor to use for the tests.
        self.z = B.as_tensor(np.array([[1., 2., 3.], [0., -1., -2.]]))

    # Tests for PointDoI.

    def test_point(self):
        res = PointDoi()(self.z)

        self.assertEqual(len(res), 1, 'PointDoi should return a single point')

        self.assertTrue(
            np.array_equal(B.as_array(res[0]), B.as_array(self.z)),
            'Value of point should not change')

    # Tests for LinearDoI.

    def test_linear(self):
        doi = LinearDoi(baseline=np.ones(B.int_shape(self.z)), resolution=21)
        res = doi(self.z)

        self.assertEqual(
            len(res), 21, 'LinearDoi should return `resolution` points')

        self.assertTrue(
            np.array_equal(B.as_array(res[0]), B.as_array(self.z)),
            'First point should be the original point')

        self.assertTrue(
            np.all(B.as_array(res[-1]) == 1.), 'Last point should be baseline')

        self.assertTrue(
            np.allclose(
                B.as_array(res[-2]),
                np.array([[1., 1.05, 1.1], [0.95, 0.9, 0.85]])),
            'Intermediate points should interpolate from baseline')

    def test_linear_point(self):
        doi = LinearDoi(resolution=1)
        res = doi(self.z)

        self.assertEqual(
            len(res), 1, 'LinearDoi should return `resolution` points')

        self.assertTrue(
            np.array_equal(B.as_array(res[0]), B.as_array(self.z)),
            'When `resolution` is 1, should be the same as PointDoi')

    def test_linear_default_baseline(self):
        doi = LinearDoi(baseline=None, resolution=10)
        res = doi(self.z)

        self.assertTrue(
            np.all(B.as_array(res[-1]) == 0.),
            'Default baseline should be zeros')

    # Tests for GaussianDoI.

    def test_gaussian(self):
        doi = GaussianDoi(var=1., resolution=10)
        res = doi(self.z)

        self.assertEqual(
            len(res), 10, 'GaussianDoi should return `resolution` points')

        self.assertEqual(B.int_shape(res[0]), B.int_shape(self.z))

    def test_gaussian_non_tensor(self):
        doi = GaussianDoi(var=1., resolution=10)
        res = doi(B.as_array(self.z))

        self.assertEqual(
            len(res), 10, 'GaussianDoi should return `resolution` points')

        self.assertEqual(res[0].shape, B.int_shape(self.z))
