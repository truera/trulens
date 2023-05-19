import numpy as np
from trulens.nn.attribution import InputAttribution
from trulens.nn.attribution import InternalInfluence
from trulens.nn.backend import get_backend
from trulens.nn.backend import tile
from trulens.nn.quantities import MaxClassQoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.nn.slices import LogitCut
from trulens.utils.typing import ModelInputs


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

        # `self.model_kwargs`` should implement exponential with five types of inputs:
        #   args[0] - (batchable) X
        #   args[1] - (batchable) Coefficients
        #   args[2] - (NOT batchable) common divisor to divide all coefficients
        #   kwargs['Degree'] - (batchable) power to raise x by
        #   kwargs['offset'] - (NOT batchable) common offset to add to all terms
        # output = X^(Degree) * Coefficients / divisor + offset
        # The first arg's first dimension determines batch_size and other batchable inputs
        #   must match this first dimension to get batched.

    # Tests for fprop.

    def test_fprop_full_network(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]]
                        ),
                    )
                ),
                np.array([5., 4., 2., 3., 3.])[:, None]
            )
        )

    def test_fprop_doi_cut(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
                        ),
                    ),
                    doi_cut=Cut(self.layer1),
                    intervention=np.array(
                        [[1., 1.], [0., 2.], [-1., 1.], [-1., 2.], [0., -1.]]
                    )
                ),
                np.array([5., 4., 0., 2., 0.])[:, None]
            )
        )

    def test_fprop_to_cut(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]]
                        ),
                    ),
                    to_cut=Cut(self.layer2)
                ), np.array([[1., 2.], [0., 2.], [0., 1.], [1., 1.], [1., 1.]])
            )
        )

    def test_fprop_identity(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
                        ),
                    ),
                    doi_cut=Cut(self.layer1),
                    to_cut=Cut(self.layer1),
                    intervention=np.array(
                        [[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]]
                    )
                ),
                np.array([[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]])
            )
        )

    def test_fprop_multiple_outputs(self):
        r = self.model.fprop(
            (np.array([[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]]),),
            to_cut=Cut([self.layer2, 'logits'])
        )

        self.assertEqual(len(r), 2)

        self.assertTrue(
            np.allclose(
                r[0],
                np.array([[1., 2.], [0., 2.], [0., 1.], [1., 1.], [1., 1.]])
            )
        )
        self.assertTrue(
            np.allclose(r[1],
                        np.array([5., 4., 2., 3., 3.])[:, None])
        )

    def test_fprop_logits_default(self):
        self.assertTrue(
            np.allclose(
                self.model.fprop(
                    (
                        np.array(
                            [[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]]
                        ),
                    ),
                    to_cut=LogitCut()
                ),
                np.array([5., 4., 2., 3., 3.])[:, None]
            )
        )

    def test_fprop_kwargs(self):
        """Test fprop on InputCut DoI for a model with both args and kwargs."""

        if not hasattr(self, 'model_kwargs'):
            # TODO: implement these tests for keras
            return

        B = get_backend()

        # Capital vars are batched, lower-case ones are not.
        X = np.array([[1., 2., 3.], [4., 5., 6.]])
        Coeffs = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
        divisor = np.array([[3.0]])
        Degree = np.array([[1., 2., 3.], [4., 5., 6.]])
        offset = np.array([[4.0]])

        actual = self.model_kwargs.fprop(
            model_args=(X, Coeffs, divisor),  # cannot swap contents
            model_kwargs=dict(Degree=Degree, offset=offset)  # between these
        )

        # Expect handling as in numpy broadcasting of the non-batched divisor, offset:
        expected = (X**Degree) * Coeffs / divisor + offset

        self.assertTrue(np.allclose(actual, expected))

    def test_fprop_kwargs_intervention(self):
        """Test fprop with InputCut and intervention/input with both args and kwargs."""

        if not hasattr(self, 'model_kwargs'):
            # TODO: implement these tests for keras
            return

        B = get_backend()

        # Capital vars are batched, lower-case ones are not.

        # batch of 2
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Coeffs = np.array([[0.5, 1.0, 1.5], [2.5, 3.0, 3.5]])
        Degree = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # non-batchable parameters
        divisor = np.array([[3.0]])
        offset = np.array([[4.0]])

        actual = self.model_kwargs.fprop(
            (X, Coeffs, divisor),               # ignored but still need to be provided
            dict(Degree=Degree, offset=offset), # ignored but still need to be provided
            doi_cut=InputCut(),
            intervention=ModelInputs(           # slightly modified model inputs
                args=(X+1.0, Coeffs+2.0, divisor+3.0),
                kwargs=dict(Degree=Degree+4.0, offset=offset+5.0)
            )
        )

        # Expect handling of the non-batched values (divisor, offset) as in
        # numpy broadcasting.
        expected = ((X + 1.0)**(Degree + 4.0)
                   ) * (Coeffs + 2.0) / (divisor + 3.0) + (offset + 5.0)

        self.assertTrue(np.allclose(actual, expected))

    def test_tile(self):
        """Test tiling utility for aligning interventions with inputs."""

        if not hasattr(self, 'model_kwargs'):
            # TODO: implement these tests for keras
            return

        B = get_backend()

        # Capital vars are batched, lower-case ones are not.

        # batch of 2
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Coeffs = np.array([[0.5, 1.0, 1.5], [2.5, 3.0, 3.5]])
        Degree = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # non-batchable parameters
        divisor = np.array([[3.0]])
        offset = np.array([[4.0]])

        # Will intervene at layer1 which is just a copy of the input X.

        # batch of 4 at intervention, Coeffs, Degree, but not (divisor, offset) should get tiled 2 times.
        X_intervention = np.array(
            [
                [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0]
            ]
        )

        # fprop no longer tiles, call the tiling method manually:
        temp_inputs = ModelInputs(
            args=[X, Coeffs, divisor],
            kwargs=dict(Degree=Degree, offset=offset)
        )
        temp_intervention = ModelInputs(args=[X_intervention])
        temp_inputs_tiled = tile(what=temp_inputs, onto=temp_intervention)

        actual = self.model_kwargs.fprop(
            temp_inputs_tiled.args,
            temp_inputs_tiled.kwargs,
            doi_cut=Cut(self.model_kwargs_layer1),
            intervention=X_intervention
        )

        # Expect handling of the non-batched values (divisor, offset) as in
        # numpy broadcasting while Degree and Coeffs should be tiled twice to
        # match intervention.
        expected = (X_intervention**np.tile(Degree, (2, 1))
                   ) * np.tile(Coeffs, (2, 1)) / divisor + offset

        self.assertTrue(np.allclose(actual, expected))

    # Tests for qoibprop.

    def test_qoibprop_full_model(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),)
                ), np.array([[3., -1.], [0., 2.]])
            )
        )

    def test_qoibprop_attribution_cut(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
                    attribution_cut=Cut(self.layer1)
                ), np.array([[3., 2.], [2., 2.]])
            )
        )

    def test_qoibprop_to_cut(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
                    to_cut=Cut(self.layer2)
                ), np.array([[1., 0.], [0., 1.]])
            )
        )

    def test_qoibprop_identity(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
                    attribution_cut=InputCut(),
                    to_cut=InputCut()
                ), np.array([[1., 0.], [0., 1.]])
            )
        )

    def test_qoibprop_internal_doi(self):
        self.assertTrue(
            np.allclose(
                self.model.qoi_bprop(
                    MaxClassQoI(), (np.array([[0., 0.], [0., 0.]]),),
                    attribution_cut=Cut(self.layer1),
                    doi_cut=Cut(self.layer1),
                    intervention=np.array([[2., 1.], [1., 2.]])
                ), np.array([[3., 2.], [3., 2.]])
            )
        )

    def test_qoibprop_multiple_inputs(self):
        r = self.model.qoi_bprop(
            MaxClassQoI(), (np.array([[2., 1.], [1., 2.]]),),
            attribution_cut=Cut([self.layer0, self.layer1])
        )

        self.assertEqual(len(r), 2)
        self.assertTrue(np.allclose(r[0], np.array([[3., -1.], [0., 2.]])))
        self.assertTrue(np.allclose(r[1], np.array([[3., 2.], [2., 2.]])))

    def test_out_cut(self):
        input_array = np.array(
            [[2., 1.], [1., 2.], [1., 1.], [1., 0.], [0., -1.]]
        )

        input_infl = InputAttribution(self.model,
                                      self.out).attributions(input_array)
        input_infl_none_out = InputAttribution(self.model
                                              ).attributions(input_array)
        np.testing.assert_array_equal(input_infl, input_infl_none_out)

        internal_infl = InternalInfluence(
            self.model, cuts=(None, self.out), qoi='max', doi='point'
        ).attributions(input_array)
        internal_infl_none_out = InternalInfluence(
            self.model, cuts=None, qoi='max', doi='point'
        ).attributions(input_array)
        np.testing.assert_array_equal(internal_infl, internal_infl_none_out)

        np.testing.assert_array_equal(internal_infl, input_infl)
