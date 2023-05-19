import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from unittest import main
from unittest import TestCase

import numpy as np
from tensorflow import Graph
import tensorflow as tf
from tensorflow.nn import relu
from trulens.nn.attribution import InternalInfluence
from trulens.nn.backend import get_backend
from trulens.nn.distributions import PointDoi
from trulens.nn.models import get_model_wrapper
from trulens.nn.quantities import ClassQoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut


class FfnEdgeCaseArchitecturesTest(TestCase):

    def test_multiple_inputs(self):
        graph = Graph()

        with graph.as_default():
            x1 = tf.placeholder('float32', (None, 5))
            z1 = x1 @ tf.random.normal((5, 6))
            x2 = tf.placeholder('float32', (None, 1))
            z2 = tf.concat([z1, x2], axis=1)
            z3 = z2 @ tf.random.normal((7, 7))
            y = z3 @ tf.random.normal((7, 3))

        model = get_model_wrapper(
            graph, input_tensors=[x1, x2], output_tensors=y
        )

        infl = InternalInfluence(model, InputCut(), ClassQoI(1), PointDoi())

        res = infl.attributions(
            [np.array([[1., 2., 3., 4., 5.]]),
             np.array([[1.]])]
        )

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 5))
        self.assertEqual(res[1].shape, (1, 1))

    def test_internal_slice_multiple_layers(self):
        graph = Graph()

        with graph.as_default():
            x1 = tf.placeholder('float32', (None, 5))
            z1 = x1 @ tf.random.normal((5, 6))
            x2 = tf.placeholder('float32', (None, 1))
            z2 = x2 @ tf.random.normal((1, 2))
            z3 = z2 @ tf.random.normal((2, 4))
            z4 = tf.concat([z1, z3], axis=1)
            z5 = z4 @ tf.random.normal((10, 7))
            y = z5 @ tf.random.normal((7, 3))

        model = get_model_wrapper(
            graph,
            input_tensors=[x1, x2],
            output_tensors=y,
            internal_tensor_dict=dict(cut_layer1=z1, cut_layer2=z2)
        )

        infl = InternalInfluence(
            model, Cut(['cut_layer1', 'cut_layer2']), ClassQoI(1), PointDoi()
        )

        res = infl.attributions(
            [np.array([[1., 2., 3., 4., 5.]]),
             np.array([[1.]])]
        )

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 6))
        self.assertEqual(res[1].shape, (1, 2))

    def test_catch_cut_name_error(self):
        graph = Graph()

        with graph.as_default():
            x = tf.placeholder('float32', (None, 2))
            z1 = x @ tf.random.normal((2, 2))
            z2 = relu(z1)
            y = z2 @ tf.random.normal((2, 1))

        model = get_model_wrapper(graph, input_tensors=x, output_tensors=y)

        with self.assertRaises(ValueError):
            infl = InternalInfluence(
                model, Cut('not_a_real_layer'), ClassQoI(0), PointDoi()
            )

            infl.attributions(np.array([[1., 1.]]))


if __name__ == '__main__':
    main()
