import numpy as np

from trulens.nn.quantities import MaxClassQoI
from trulens.nn.slices import Cut, InputCut, LogitCut


class ModelWrapperTestBase(object):

    def setUp(self):
        # `self.model` should be implemented in the respective backend to match
        # the following network:
        #
        # Output:        o        name: 'logits'
        #              /   \
        #           1 /     \ 2
        # Layer 2:   o       o    cut identifier: `self.layer2`
        #            | \   / |
        #            |  \ /  |
        #            |   x   |
        #            | 1/ \0 |
        #          1 | /   \ | 1
        # Layer 1:   o       o    cut identifier: `self.layer1`
        #            | \   / |
        #            |  \ /  |
        #            |   x   |
        #            | 0/ \-1|
        #          1 | /   \ | 1
        # Input:     o       o    cut identifier: `self.layer0`

        self.layer1_weights = np.array([[1., 0.], [-1., 1.]])
        self.layer2_weights = np.array([[1., 1.], [0., 1.]])
        self.layer3_weights = np.array([[1.], [2.]])
        self.internal_bias = np.array([0., 0.])
        self.bias = np.array([0.])

    # Tests for fprop.

    def test_fprop_full_network(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[2., 1.], [1., 2.], [1., 1.], [1., 0.],
                             [0., -1.]]),))[0],
                np.array([5., 4., 2., 3., 3.])[:, None]))

    def test_fprop_doi_cut(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]
                            ]),),
                    doi_cut=Cut(self.layer1),
                    intervention=np.array(
                        [[1., 1.], [0., 2.], [-1., 1.], [-1., 2.], [0.,
                                                                    -1.]]))[0],
                np.array([5., 4., 0., 2., 0.])[:, None]))

    def test_fprop_to_cut(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]
                            ]),),
                    to_cut=Cut(self.layer2))[0],
                np.array([[1., 2.], [0., 2.], [0., 1.], [1., 1.], [1., 1.]])))

    def test_fprop_identity(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]
                            ]),),
                    doi_cut=Cut(self.layer1),
                    to_cut=Cut(self.layer1),
                    intervention=np.array(
                        [[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0.,
                                                                  -1.]]))[0],
                np.array([[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]])))

    def test_fprop_multiple_outputs(self):
        r = self.model.fprop(
            (np.array([[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]]),),
            to_cut=Cut([self.layer2, 'logits']))

        self.assertEqual(len(r), 2)

        self.assertTrue(
            np.allclose(
                r[0],
                np.array([[1., 2.], [0., 2.], [0., 1.], [1., 1.], [1., 1.]])))
        self.assertTrue(
            np.allclose(r[1],
                        np.array([5., 4., 2., 3., 3.])[:, None]))

    def test_fprop_logits_default(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]
                            ]),),
                    to_cut=LogitCut())[0],
                np.array([5., 4., 2., 3., 3.])[:, None]))

    # Tests for qoibprop.

    def test_qoibprop_full_model(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),)),
                np.array([[3., -1.], [0., 2.]])))

    def test_qoibprop_attribution_cut(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
                    attribution_cut=Cut(self.layer1)),
                np.array([[3., 2.], [2., 2.]])))

    def test_qoibprop_to_cut(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
                    to_cut=Cut(self.layer2)), np.array([[1., 0.], [0., 1.]])))

    def test_qoibprop_identity(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
                    attribution_cut=InputCut(),
                    to_cut=InputCut()), np.array([[1., 0.], [0., 1.]])))

    def test_qoibprop_internal_doi(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[0., 0.], [0., 0.]]),),
                    attribution_cut=Cut(self.layer1),
                    doi_cut=Cut(self.layer1),
                    intervention=np.array([[2., 1.], [1., 2.]])),
                np.array([[3., 2.], [3., 2.]])))

    def test_qoibprop_multiple_inputs(self):
        r = self.model.qoi_bprop(
            MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
            attribution_cut=Cut([self.layer0, self.layer1]))

        self.assertEqual(len(r), 2)
        self.assertTrue(np.allclose(r[0], np.array([[3., -1.], [0., 2.]])))
        self.assertTrue(np.allclose(r[1], np.array([[3., 2.], [2., 2.]])))
