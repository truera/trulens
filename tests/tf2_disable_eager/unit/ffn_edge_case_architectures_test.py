import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import TestCase, main

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Input, Concatenate
from tensorflow.keras.models import Model

from trulens.nn import backend as B
from trulens.nn.attribution import InternalInfluence
from trulens.nn.distributions import PointDoi
from trulens.nn.models import ModelWrapper
from trulens.nn.quantities import ClassQoI
from trulens.nn.slices import InputCut, Cut

assert (not tf.executing_eagerly())


class FfnEdgeCaseArchitecturesTest(TestCase):

    def test_multiple_inputs(self):
        x1 = Input((5,))
        z1 = Dense(6)(x1)
        x2 = Input((1,))
        z2 = Concatenate()([z1, x2])
        z3 = Dense(7)(z2)
        y = Dense(3)(z3)

        model = ModelWrapper(Model([x1, x2], y))

        infl = InternalInfluence(model, InputCut(), ClassQoI(1), PointDoi())

        res = infl.attributions(
            [np.array([[1., 2., 3., 4., 5.]]),
             np.array([[1.]])])

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 5))
        self.assertEqual(res[1].shape, (1, 1))

    def test_multiple_outputs(self):
        x = Input((5,))
        z1 = Dense(6)(x)
        z2 = Dense(7)(z1)
        y1 = Dense(2)(z2)
        z3 = Dense(8)(z2)
        y2 = Dense(3)(z3)

        model = ModelWrapper(Model(x, [y1, y2]))

        # TODO(klas): how should we handle these types of models?

    def test_internal_multiple_inputs(self):
        x1 = Input((5,))
        z1 = Dense(6)(x1)
        x2 = Input((1,))
        z2 = Concatenate(name='concat')([z1, x2])
        z3 = Dense(7)(z2)
        y = Dense(3)(z3)

        model = ModelWrapper(Model([x1, x2], y))

        infl = InternalInfluence(
            model, Cut('concat', anchor='in'), ClassQoI(1), PointDoi())

        res = infl.attributions(
            [np.array([[1., 2., 3., 4., 5.]]),
             np.array([[1.]])])

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 6))
        self.assertEqual(res[1].shape, (1, 1))

    def test_internal_slice_multiple_layers(self):
        x1 = Input((5,))
        z1 = Dense(6, name='cut_layer1')(x1)
        x2 = Input((1,))
        z2 = Dense(2, name='cut_layer2')(x2)
        z3 = Dense(4)(z2)
        z4 = Concatenate()([z1, z3])
        z5 = Dense(7)(z4)
        y = Dense(3)(z5)

        model = ModelWrapper(Model([x1, x2], y))

        infl = InternalInfluence(
            model, Cut(['cut_layer1', 'cut_layer2']), ClassQoI(1), PointDoi())

        res = infl.attributions(
            [np.array([[1., 2., 3., 4., 5.]]),
             np.array([[1.]])])

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 6))
        self.assertEqual(res[1].shape, (1, 2))

    def test_anchors(self):
        x = Input((2,))
        z1 = Dense(2)(x)
        z2 = Activation('relu')(z1)
        y = Dense(1)(z2)

        k_model = Model(x, y)
        k_model.set_weights(
            [
                np.array([[1., 0.], [0., -1.]]),
                np.array([0., 0.]),
                np.array([[1.], [1.]]),
                np.array([0.])
            ])

        model = ModelWrapper(k_model)

        infl_out = InternalInfluence(
            model,
            Cut(2, anchor='out'),
            ClassQoI(0),
            PointDoi(),
            multiply_activation=False)

        infl_in = InternalInfluence(
            model,
            Cut(2, anchor='in'),
            ClassQoI(0),
            PointDoi(),
            multiply_activation=False)

        res_out = infl_out.attributions(np.array([[1., 1.]]))
        res_in = infl_in.attributions(np.array([[1., 1.]]))

        self.assertEqual(res_out.shape, (1, 2))
        self.assertEqual(res_in.shape, (1, 2))
        self.assertTrue(np.allclose(res_out, np.array([[1., 1.]])))
        self.assertTrue(np.allclose(res_in, np.array([[1., 0.]])))

    def test_catch_cut_index_error(self):
        x = Input((2,))
        z1 = Dense(2)(x)
        z2 = Activation('relu')(z1)
        y = Dense(1)(z2)

        model = ModelWrapper(Model(x, y))

        with self.assertRaises(ValueError):
            infl = InternalInfluence(model, Cut(4), ClassQoI(0), PointDoi())

            infl.attributions(np.array([[1., 1.]]))

    def test_catch_cut_name_error(self):
        x = Input((2,))
        z1 = Dense(2)(x)
        z2 = Activation('relu')(z1)
        y = Dense(1, name='logits')(z2)

        model = ModelWrapper(Model(x, y))

        with self.assertRaises(ValueError):
            infl = InternalInfluence(
                model, Cut('not_a_real_layer'), ClassQoI(0), PointDoi())

            infl.attributions(np.array([[1., 1.]]))


if __name__ == '__main__':
    main()
