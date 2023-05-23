import numpy as np
from trulens.nn.backend import get_backend
from trulens.nn.quantities import *


class QoiTestBase(object):

    def setUp(self):
        self.B = get_backend()
        # Create an example tensor to use for the tests.
        self.y = self.B.as_tensor(np.array([[1., 2., 3.], [0., -1., -2.]]))
        self.z = self.B.as_tensor(
            np.array(
                [
                    [
                        [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 0., 1.]],
                        [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]]
                    ],
                    [
                        [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 0., 1., 2.]],
                        [[0., 2., 4., 6.], [0., 3., 6., 9.], [0., 1., 2., 3.]]
                    ]
                ]
            )
        )

    # Tests for MaxClassQoI.

    def test_max_class_no_activation(self):
        qoi = MaxClassQoI()
        res = qoi(self.y)

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([3., 0.])))

    def test_max_class_axis(self):
        qoi = MaxClassQoI(axis=0)
        res = qoi(self.y)

        self.assertTrue(
            np.allclose(self.B.as_array(res), np.array([1., 2., 3.]))
        )

    def test_max_class_activation_string(self):
        qoi = MaxClassQoI(activation='softmax')
        res = qoi(self.y)

        self.assertTrue(
            np.allclose(
                self.B.as_array(res), np.array([0.66524096, 0.66524096])
            )
        )

    def test_max_class_activation_function(self):
        qoi = MaxClassQoI(activation=self.B.softmax)
        res = qoi(self.y)

        self.assertTrue(
            np.allclose(
                self.B.as_array(res), np.array([0.66524096, 0.66524096])
            )
        )

    # Tests for InternalChannelQoI.

    def test_internal_channel_1d(self):
        qoi = InternalChannelQoI(2)
        res = qoi(self.y)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([3., -2.])))

    def test_internal_channel_axis1(self):
        qoi = InternalChannelQoI(1, channel_axis=1)
        res = qoi(self.z)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([24., 36.])))

    def test_internal_channel_axis3(self):
        qoi = InternalChannelQoI(1, channel_axis=3)
        res = qoi(self.z)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([21., 14.])))

    def test_internal_channel_invalid_channel(self):
        qoi = InternalChannelQoI(0, channel_axis=0)
        with self.assertRaises(ValueError):
            qoi(self.z)

    # Tests for ClassQoI.

    def test_class(self):
        qoi = ClassQoI(1)
        res = qoi(self.y)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([2., -1.])))

    # Tests for ComparativeQoI.

    def test_comparative(self):
        qoi = ComparativeQoI(1, 0)
        res = qoi(self.y)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([1., -1.])))

    # Tests for LambdaQoI.

    def test_lambda(self):
        qoi = LambdaQoI(lambda y: y[:, 0] + y[:, 1])
        res = qoi(self.y)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([3., -1.])))

    def test_lambda_error(self):
        with self.assertRaises(ValueError):
            # Should only accept functions taking one argument.
            qoi = LambdaQoI(lambda y1, y2: y[:, 0] + y2[:, 1])

    # Tests for ThresholdQoI.

    def test_threshold_high_minus_low(self):
        qoi = ThresholdQoI(1.5)
        res = qoi(self.y)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([4., 3.])))

    def test_threshold_low_minus_high(self):
        qoi = ThresholdQoI(1.5, low_minus_high=True)
        res = qoi(self.y)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(np.allclose(self.B.as_array(res), np.array([-4., -3.])))

    def test_threshold_activation(self):
        qoi = ThresholdQoI(0.75, activation='sigmoid')
        res = qoi(self.y)

        self.assertEqual(
            self.B.int_shape(res), (2,),
            'Should return one scalar per row in the batch'
        )

        self.assertTrue(
            np.allclose(
                self.B.as_array(res), np.array([1.1023126, -0.8881443])
            )
        )
