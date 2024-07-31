import os

os.environ['TRULENS_BACKEND'] = 'pytorch'

from unittest import main
from unittest import TestCase

import numpy as np
from torch import cat
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from trulens.nn.attribution import InternalInfluence
from trulens.nn.backend import get_backend
from trulens.nn.distributions import PointDoi
from trulens.nn.models import get_model_wrapper
from trulens.nn.quantities import ClassQoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut


class FfnEdgeCaseArchitecturesTest(TestCase):

    def test_multiple_inputs(self):

        class M(Module):

            def __init__(this):
                super(M, this).__init__()
                this.z1 = Linear(5, 6)
                this.z3 = Linear(7, 7)
                this.y = Linear(7, 3)

            def forward(this, x1, x2):
                x1 = this.z1(x1)
                z = cat((x1, x2), 1)
                z = this.z3(z)
                return this.y(z)

        model = get_model_wrapper(M())

        infl = InternalInfluence(model, InputCut(), ClassQoI(1), PointDoi())

        res = infl.attributions(
            np.array([[1., 2., 3., 4., 5.]]).astype('float32'),
            np.array([[1.]]).astype('float32')
        )

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 5))
        self.assertEqual(res[1].shape, (1, 1))

    def test_internal_multiple_inputs(self):

        class ConcatenateLayer(Module):

            def forward(this, x1, x2):
                return cat((x1, x2), 1)

        class M(Module):

            def __init__(this):
                super(M, this).__init__()
                this.z1 = Linear(5, 6)
                this.concat = ConcatenateLayer()
                this.z3 = Linear(7, 7)
                this.y = Linear(7, 3)

            def forward(this, x1, x2):
                x1 = this.z1(x1)
                z = this.concat(x1, x2)
                z = this.z3(z)
                return this.y(z)

        model = get_model_wrapper(M())

        infl = InternalInfluence(
            model, Cut('concat', anchor='in'), ClassQoI(1), PointDoi()
        )

        res = infl.attributions(
            np.array([[1., 2., 3., 4., 5.]]).astype('float32'),
            np.array([[1.]]).astype('float32')
        )

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 6))
        self.assertEqual(res[1].shape, (1, 1))

    def test_internal_slice_multiple_layers(self):

        class M(Module):

            def __init__(this):
                super(M, this).__init__()
                this.cut_layer1 = Linear(5, 6)
                this.cut_layer2 = Linear(1, 2)
                this.z3 = Linear(2, 4)
                this.z5 = Linear(10, 7)
                this.y = Linear(7, 3)

            def forward(this, x1, x2):
                z1 = this.cut_layer1(x1)
                z2 = this.cut_layer2(x2)
                z3 = this.z3(z2)
                z4 = cat((z1, z3), 1)
                z5 = this.z5(z4)
                return this.y(z5)

        model = get_model_wrapper(M())

        infl = InternalInfluence(
            model, Cut(['cut_layer1', 'cut_layer2']), ClassQoI(1), PointDoi()
        )

        res = infl.attributions(
            np.array([[1., 2., 3., 4., 5.]]).astype('float32'),
            np.array([[1.]]).astype('float32')
        )

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 6))
        self.assertEqual(res[1].shape, (1, 2))

    def test_anchors(self):

        class M(Module):

            def __init__(this):
                super(M, this).__init__()
                this.z1 = Linear(2, 2)
                this.z2 = ReLU()
                this.y = Linear(2, 1)

                B = get_backend()
                this.z1.weight.data = B.as_tensor(
                    np.array([[1., 0.], [0., -1.]]).T
                )
                this.z1.bias.data = B.as_tensor(np.array([0., 0.]))
                this.y.weight.data = B.as_tensor(np.array([[1.], [1.]]).T)
                this.y.bias.data = B.as_tensor(np.array([0.]))

            def forward(this, x):
                z1 = this.z1(x)
                z2 = this.z2(z1)
                return this.y(z2)

        model = get_model_wrapper(M())

        infl_out = InternalInfluence(
            model,
            Cut('z2', anchor='out'),
            ClassQoI(0),
            PointDoi(),
            multiply_activation=False
        )

        infl_in = InternalInfluence(
            model,
            Cut('z2', anchor='in'),
            ClassQoI(0),
            PointDoi(),
            multiply_activation=False
        )

        res_out = infl_out.attributions(np.array([[1., 1.]]))
        res_in = infl_in.attributions(np.array([[1., 1.]]))

        self.assertEqual(res_out.shape, (1, 2))
        self.assertEqual(res_in.shape, (1, 2))
        self.assertTrue(np.allclose(res_out, np.array([[1., 1.]])))
        self.assertTrue(np.allclose(res_in, np.array([[1., 0.]])))

    def test_catch_cut_name_error(self):

        class M(Module):

            def __init__(this):
                super(M, this).__init__()
                this.z1 = Linear(2, 2)
                this.z2 = ReLU()
                this.y = Linear(2, 1)

            def forward(this, x):
                z1 = this.z1(x)
                z2 = this.z2(z1)
                return this.y(z2)

        model = get_model_wrapper(M())

        with self.assertRaises(ValueError):
            infl = InternalInfluence(
                model, Cut('not_a_real_layer'), ClassQoI(0), PointDoi()
            )

            infl.attributions(np.array([[1., 1.]]).astype('float32'))


if __name__ == '__main__':
    main()
