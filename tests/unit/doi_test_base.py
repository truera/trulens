import numpy as np

from trulens.nn.backend import get_backend
from trulens.nn.distributions import PointDoi, LinearDoi, GaussianDoi


class DoiTestBase(object):

    def setUp(self):
        # Create an example tensor to use for the tests.
        self.B = get_backend()
        self.z = self.B.as_tensor(np.array([[1., 2., 3.], [0., -1., -2.]]))

    # Tests for PointDoI.

    def test_point(self):

        res = PointDoi()(self.z)

        self.assertEqual(len(res), 1, 'PointDoi should return a single point')

        self.assertTrue(
            np.array_equal(self.B.as_array(res[0]), self.B.as_array(self.z)),
            'Value of point should not change')

    # Tests for LinearDoI.

    def test_linear(self):

        doi = LinearDoi(
            baseline=np.ones(self.B.int_shape(self.z)), resolution=21)
        res = doi(self.z)

        self.assertEqual(
            len(res), 21, 'LinearDoi should return `resolution` points')

        self.assertTrue(
            np.array_equal(self.B.as_array(res[0]), self.B.as_array(self.z)),
            'First point should be the original point')

        self.assertTrue(
            np.all(self.B.as_array(res[-1]) == 1.),
            'Last point should be baseline')

        self.assertTrue(
            np.allclose(
                self.B.as_array(res[-2]),
                np.array([[1., 1.05, 1.1], [0.95, 0.9, 0.85]])),
            'Intermediate points should interpolate from baseline')

    def test_linear_point(self):
        doi = LinearDoi(resolution=1)
        res = doi(self.z)

        self.assertEqual(
            len(res), 1, 'LinearDoi should return `resolution` points')

        self.assertTrue(
            np.array_equal(self.B.as_array(res[0]), self.B.as_array(self.z)),
            'When `resolution` is 1, should be the same as PointDoi')

    def test_linear_default_baseline(self):
        doi = LinearDoi(baseline=None, resolution=10)
        res = doi(self.z)

        self.assertTrue(
            np.all(self.B.as_array(res[-1]) == 0.),
            'Default baseline should be zeros')

    def test_linear_from_computed_nearby_baseline(self):
        # Baseline for cut value is a function of value at that cut.

        doi = LinearDoi(baseline=lambda z, model_inputs: z + 42)

        res = doi(self.z)

        self.assertTrue(
            np.all(self.B.as_array(res[0]) == self.B.as_array(self.z)),
            'Starting point of linear baseline should be base cut value.')

        self.assertTrue(
            np.all(self.B.as_array(res[-1]) == self.B.as_array(self.z + 42)),
            'End point of linear baseline should be computed cut value.')

    # Tests for GaussianDoI.

    def test_gaussian(self):

        doi = GaussianDoi(var=1., resolution=10)
        res = doi(self.z)

        self.assertEqual(
            len(res), 10, 'GaussianDoi should return `resolution` points')

        self.assertEqual(self.B.int_shape(res[0]), self.B.int_shape(self.z))

    def test_gaussian_non_tensor(self):

        doi = GaussianDoi(var=1., resolution=10)
        res = doi(self.B.as_array(self.z))

        self.assertEqual(
            len(res), 10, 'GaussianDoi should return `resolution` points')

        self.assertEqual(res[0].shape, self.B.int_shape(self.z))
