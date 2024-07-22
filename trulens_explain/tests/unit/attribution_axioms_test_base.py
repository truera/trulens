'''
These unit tests check that various parameterizations of Internal Influence
satisfy several axioms from the following two papers:

    1.  Leino et al. "Influence-based Explanations for Convolutional Neural
        Networks" ITC 2018 [Arxiv](https://arxiv.org/pdf/1802.03788.pdf)

    2.  Sundararajan et al. "Axiomatic Attribution for Deep Networks" ICML 2017
        [ArXiv](https://arxiv.org/pdf/1703.01365.pdf)

These axioms should hold on arbitrary networks.
'''

from functools import partial

import numpy as np
from trulens.nn.attribution import InternalInfluence
from trulens.nn.backend import get_backend
from trulens.nn.distributions import DoI
from trulens.nn.distributions import LinearDoi
from trulens.nn.distributions import PointDoi
from trulens.nn.quantities import ClassQoI
from trulens.nn.quantities import InternalChannelQoI
from trulens.nn.quantities import MaxClassQoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.utils.test import tolerance


class AxiomsTestBase(object):

    def setUp(self):
        self.atol = tolerance(get_backend())
        print('atol=', self.atol)

        np.random.seed(2020)

        self.input_size = 5
        self.internal1_size = 10
        self.internal2_size = 8
        self.output_size = 3

        # Make weights for a linear model for testing.
        self.model_lin_weights = np.random.normal(
            scale=2. / (self.input_size + self.output_size),
            size=(self.input_size, self.output_size)
        )
        self.model_lin_bias = np.random.uniform(-0.5, 0.5, (self.output_size,))

        # NOTE: subclass should add `self.model_lin`.

        # Make weights for a deeper model for testing.
        self.model_deep_weights_1 = np.random.normal(
            scale=2. / (self.input_size + self.internal1_size),
            size=(self.input_size, self.internal1_size)
        )
        self.model_deep_bias_1 = np.random.uniform(
            -0.5, 0.5, (self.internal1_size,)
        )
        self.model_deep_weights_2 = np.random.normal(
            scale=2. / (self.internal1_size + self.internal2_size),
            size=(self.internal1_size, self.internal2_size)
        )
        self.model_deep_bias_2 = np.random.uniform(
            -0.5, 0.5, (self.internal2_size,)
        )
        self.model_deep_weights_3 = np.random.normal(
            scale=2. / (self.internal2_size + self.output_size),
            size=(self.internal2_size, self.output_size)
        )
        self.model_deep_bias_3 = np.random.uniform(
            -0.5, 0.5, (self.output_size,)
        )

        # NOTE: subclass should add `self.model_deep`, `self.layer2`, and
        #   `self.layer3`.

        # Make a test data point.
        self.x = np.array([[1., 2., 3., 4., 5.], [0., -1., -2., 2., 1.]])

        self.baseline = np.array([[1., 2., 3., -2., 5.]])

    def test_idempotence(self):
        infl = InternalInfluence(
            self.model_lin,
            InputCut(),
            MaxClassQoI(),
            PointDoi(),
            multiply_activation=False
        )

        res1 = infl.attributions(self.x)
        res2 = infl.attributions(self.x)

        self.assertTrue(np.allclose(res1, res2))

        infl_act = InternalInfluence(
            self.model_lin,
            InputCut(),
            MaxClassQoI(),
            PointDoi(),
            multiply_activation=True
        )

        res1 = infl_act.attributions(self.x)
        res2 = infl_act.attributions(self.x)

        self.assertTrue(np.allclose(res1, res2))

    # Tests for linear agreement [1].
    #
    # This axiom states that if the model is linear, the attribution should
    # simply be the weights of the model.

    def test_linear_agreement(self):
        c = 1
        infl = InternalInfluence(
            self.model_lin,
            InputCut(),
            ClassQoI(c),
            PointDoi(),
            multiply_activation=False
        )

        res = infl.attributions(self.x)

        self.assertEqual(res.shape, (2, self.input_size))

        self.assertTrue(
            np.allclose(res[0], self.model_lin_weights[:, c], atol=self.atol)
        )
        self.assertTrue(
            np.allclose(res[1], self.model_lin_weights[:, c], atol=self.atol)
        )

    def test_linear_agreement_multiply_activation(self):
        c = 1
        infl = InternalInfluence(
            self.model_lin,
            InputCut(),
            ClassQoI(c),
            PointDoi(),
            multiply_activation=True
        )

        res = infl.attributions(self.x)

        self.assertEqual(res.shape, (2, self.input_size))

        self.assertTrue(
            np.allclose(
                res, self.model_lin_weights[:, c] * self.x, atol=self.atol
            )
        )

    def test_linear_agreement_linear_slice(self):
        c = 4
        infl = InternalInfluence(
            self.model_deep, (Cut(self.layer2), Cut(self.layer3)),
            InternalChannelQoI(c),
            PointDoi(),
            multiply_activation=False
        )

        res = infl.attributions(self.x)

        self.assertEqual(res.shape, (2, self.internal1_size))

        self.assertTrue(
            np.allclose(
                res[0], self.model_deep_weights_2[:, c], atol=self.atol
            )
        )
        self.assertTrue(
            np.allclose(
                res[1], self.model_deep_weights_2[:, c], atol=self.atol
            )
        )

    def test_linear_agreement_linear_slice_multiply_activation(self):
        c = 4
        infl = InternalInfluence(
            self.model_deep, (Cut(self.layer2), Cut(self.layer3)),
            InternalChannelQoI(c),
            PointDoi(),
            multiply_activation=True
        )

        res = infl.attributions(self.x)

        self.assertEqual(res.shape, (2, self.internal1_size))

        z = self.model_deep.fprop((self.x,), to_cut=Cut(self.layer2))

        self.assertTrue(
            np.allclose(
                res, self.model_deep_weights_2[:, c] * z, atol=self.atol
            )
        )

    # Tests for sensitivity [2].
    #
    # This axiom states that if the baseline and the input differ in value for
    # exactly one variable, and the output of the model on the baseline is
    # different from on the input, then the differing variable must be given
    # non-zero attribution.

    def test_sensitivity(self):
        c = 2
        infl = InternalInfluence(
            self.model_deep,
            InputCut(),
            ClassQoI(c),
            LinearDoi(self.baseline),
            multiply_activation=False
        )

        out_x = self.model_deep.fprop((self.x[0:1],))[:, c]
        out_baseline = self.model_deep.fprop((self.baseline,))[:, c]

        if not np.allclose(out_x, out_baseline):
            res = infl.attributions(self.x)

            self.assertEqual(res.shape, (2, self.input_size))

            self.assertNotEqual(res[0, 3], 0.)

    # Tests for distributional linearity [1].
    #
    # This axiom states that the attribution should be linear over the
    # distribution of interest, i.e., the total attribution is the sum of the
    # point-wise attribution weighted by the probability of each point.

    def test_distributional_linearity(self):
        x1, x2 = self.x[0:1], self.x[1:]
        p1, p2 = 0.25, 0.75

        class DistLinDoI(DoI):
            '''
            Represents the distribution of interest that weights `z` with
            probability 1/4 and `z + diff` with probability 3/4.
            '''

            def __init__(self, diff):
                super(DistLinDoI, self).__init__()
                self.diff = diff

            def __call__(self, z):
                return [z, z + self.diff, z + self.diff, z + self.diff]

        infl_pt = InternalInfluence(
            self.model_deep,
            InputCut(),
            ClassQoI(0),
            PointDoi(),
            multiply_activation=False
        )

        attr1 = infl_pt.attributions(x1)
        attr2 = infl_pt.attributions(x2)

        infl_dl = InternalInfluence(
            self.model_deep,
            InputCut(),
            ClassQoI(0),
            DistLinDoI(x2 - x1),
            multiply_activation=False
        )

        attr12 = infl_dl.attributions(x1)

        self.assertTrue(np.allclose(attr12, p1 * attr1 + p2 * attr2))

    def test_distributional_linearity_internal_influence(self):
        x1, x2 = self.x[0:1], self.x[1:]
        p1, p2 = 0.25, 0.75

        class DistLinDoI(DoI):
            '''
            Represents the distribution of interest that weights `z` with
            probability 1/4 and `z + diff` with probability 3/4.
            '''

            def __init__(self, diff):
                super(DistLinDoI, self).__init__()
                self.diff = diff

            def __call__(self, z):
                return [z, z + self.diff, z + self.diff, z + self.diff]

        infl_pt = InternalInfluence(
            self.model_deep,
            Cut(self.layer2),
            ClassQoI(0),
            PointDoi(),
            multiply_activation=False
        )

        attr1 = infl_pt.attributions(x1)
        attr2 = infl_pt.attributions(x2)

        infl_dl = InternalInfluence(
            self.model_deep,
            Cut(self.layer2),
            ClassQoI(0),
            DistLinDoI(x2 - x1),
            multiply_activation=False
        )

        attr12 = infl_dl.attributions(x1)

        self.assertTrue(np.allclose(attr12, p1 * attr1 + p2 * attr2))

    # Tests for completeness [2].
    #
    # This axiom states that (when the distribution is `LinearDoI` and
    # `multiply_activation` is True) the sum of the attributions must equal the
    # difference in ouput between the point and the baseline.

    def test_completeness(self):
        c = 2
        infl = InternalInfluence(
            self.model_deep,
            InputCut(),
            ClassQoI(c),
            LinearDoi(self.baseline, resolution=100),
            multiply_activation=True
        )

        out_x = self.model_deep.fprop((self.x,))[:, c]
        out_baseline = self.model_deep.fprop((self.baseline,))[:, c]

        res = infl.attributions(self.x)

        self.assertTrue(
            np.allclose(res.sum(axis=1), out_x - out_baseline, atol=5e-2)
        )

    def test_completeness_zero_baseline(self):
        c = 2
        infl = InternalInfluence(
            self.model_deep,
            InputCut(),
            ClassQoI(c),
            LinearDoi(resolution=100),
            multiply_activation=True
        )

        out_x = self.model_deep.fprop((self.x,))[:, c]
        out_baseline = self.model_deep.fprop((self.baseline * 0,))[:, c]

        res = infl.attributions(self.x)

        self.assertTrue(
            np.allclose(res.sum(axis=1), out_x - out_baseline, atol=5e-2)
        )

    def test_completeness_internal(self):
        c = 2

        baseline = np.array([[0., 1., 2., 3., 4., 3., 2., 1., 0., -1.]])

        infl = InternalInfluence(
            self.model_deep,
            Cut(self.layer2),
            ClassQoI(c),
            LinearDoi(baseline, resolution=100, cut=Cut(self.layer2)),
            multiply_activation=True
        )

        g = partial(
            self.model_deep.fprop,
            doi_cut=Cut(self.layer2),
            intervention=np.tile(baseline, (2, 1))
        )

        out_x = self.model_deep.fprop((self.x,))[:, c]
        out_baseline = g((self.x,))[:, c]

        res = infl.attributions(self.x)

        self.assertTrue(
            np.allclose(res.sum(axis=1), out_x - out_baseline, atol=5e-2)
        )

    def test_completeness_internal_zero_baseline(self):
        c = 2

        infl = InternalInfluence(
            self.model_deep,
            Cut(self.layer2),
            ClassQoI(c),
            LinearDoi(resolution=100, cut=Cut(self.layer2)),
            multiply_activation=True
        )

        g = partial(
            self.model_deep.fprop,
            doi_cut=Cut(self.layer2),
            intervention=np.zeros((2, 10))
        )
        out_x = self.model_deep.fprop((self.x,))[:, c]
        out_baseline = g((self.x,))[:, c]

        res = infl.attributions(self.x)

        self.assertTrue(
            np.allclose(res.sum(axis=1), out_x - out_baseline, atol=5e-2)
        )
