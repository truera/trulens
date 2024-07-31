import numpy as np
from trulens.nn.attribution import InternalInfluence
from trulens.nn.distributions import LinearDoi
from trulens.nn.quantities import MaxClassQoI
from trulens.nn.slices import InputCut


class BatchTestBase(object):

    def setUp(self):
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

        # NOTE: subclass should add `self.model_deep`.

        # Make a test data point.
        self.batch_x = np.random.uniform(size=(5, self.input_size))

    def test_batch_processing_lin(self):
        infl = InternalInfluence(
            self.model_lin, InputCut(), MaxClassQoI(), LinearDoi()
        )

        r1 = np.concatenate([infl.attributions(x[None]) for x in self.batch_x])
        r2 = infl.attributions(self.batch_x)

        self.assertTrue(np.allclose(r1, r2))

    def test_batch_processing_deep(self):
        infl = InternalInfluence(
            self.model_deep, InputCut(), MaxClassQoI(), LinearDoi()
        )

        r1 = np.concatenate([infl.attributions(x[None]) for x in self.batch_x])
        r2 = infl.attributions(self.batch_x)

        self.assertTrue(np.allclose(r1, r2))
