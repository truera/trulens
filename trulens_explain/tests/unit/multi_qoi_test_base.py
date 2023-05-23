from unittest import main
from unittest import TestCase

import numpy as np
from trulens.nn.attribution import InternalInfluence
from trulens.nn.backend import get_backend
from trulens.nn.distributions import DoI
from trulens.nn.quantities import QoI
from trulens.nn.slices import Cut
from trulens.utils.typing import OM
from trulens.utils.typing import Outputs
from trulens.utils.typing import Tensor


class PerTimestepQoI(QoI):

    def __call__(self, x):
        if isinstance(x, tuple):
            x = x[0]  # torch RNN layer outputs tuple of (vals, hidden_states)
        num_classes = x.shape[-1]
        num_timesteps = x.shape[1]
        out = []
        for i in range(num_timesteps):
            for j in range(num_classes):
                out.append(x[:, i, j])
        return out


class RNNLinearDoi(DoI):

    def __init__(self, baseline=None, resolution=10, cut: Cut = None):
        """
        __init__ Constructor

        Parameters
        ----------
        baseline : backend.Tensor
            Must be same shape as support, i.e., shape of z points
            eventually passed to __call__
        resolution : int
            Number of points returned by each call to this DoI
        """
        super(RNNLinearDoi, self).__init__(cut)
        self._baseline = baseline
        self._resolution = resolution
        self.B = get_backend()

    def calc_doi(self, x_input, tf_cell=False) -> OM[Outputs, Tensor]:

        x = x_input[0] if tf_cell else x_input
        batch_size = len(x)

        if self._baseline is None:
            if self.B.is_tensor(x):
                x = self.B.as_array(x)
            baseline = np.zeros_like(x)
        else:
            baseline = self._baseline

        tile_dims = [1] * len(baseline.shape)
        tile_dims[0] = batch_size
        baseline = baseline[0, ...]
        baseline = np.tile(baseline, tuple(tile_dims))

        if self.B.is_tensor(x) and not self.B.is_tensor(baseline):
            baseline = self.B.as_tensor(baseline)

        if not self.B.is_tensor(x) and self.B.is_tensor(baseline):
            baseline = self.B.as_array(baseline)

        r = self._resolution - 1.
        doi_out = [
            (1. - i / r) * x + i / r * baseline
            for i in range(self._resolution)
        ]
        if tf_cell:
            doi_out = [[d, x_input[1]] for d in doi_out]
        return doi_out

    def __call__(self, x) -> OM[Outputs, Tensor]:
        return self.calc_doi(x, tf_cell=False)

    def get_activation_multiplier(self, activation):

        batch_size = len(activation)
        if (self._baseline is None):
            baseline = np.zeros_like(activation)
        else:
            baseline = self._baseline

        tile_dims = [1] * len(baseline.shape)
        tile_dims[0] = batch_size
        baseline = baseline[0, ...]
        baseline = np.tile(baseline, tuple(tile_dims))
        if (self.B.is_tensor(activation) and not self.B.is_tensor(baseline)):
            baseline = self.B.as_tensor(baseline)

        if (not self.B.is_tensor(activation) and self.B.is_tensor(baseline)):
            baseline = self.B.as_array(baseline)

        batch_size = len(activation)
        return activation - baseline


class MultiQoiTestBase(TestCase):

    def per_timestep_qoi(
        self, model_wrapper, num_classes, num_features, num_timesteps,
        batch_size
    ):
        cuts = (Cut('rnn', 'in', None), Cut('dense', 'out', None))

        infl = InternalInfluence(
            model_wrapper, cuts, PerTimestepQoI(), RNNLinearDoi()
        )

        input_attrs = infl.attributions(
            np.ones((batch_size, num_timesteps, num_features)
                   ).astype('float32')
        )

        original_output_shape = (
            num_classes * num_timesteps, batch_size, num_timesteps, num_features
        )
        self.assertEqual(np.stack(input_attrs).shape, original_output_shape)

        rotated = np.stack(input_attrs, axis=-1)
        attr_shape = list(rotated.shape)[:-1]
        attr_shape.append(int(rotated.shape[-1] / num_classes))
        attr_shape.append(num_classes)
        attr_shape = tuple(attr_shape)
        self.assertEqual(
            attr_shape, (
                batch_size, num_timesteps, num_features, num_timesteps,
                num_classes
            )
        )
        input_attrs = np.reshape(rotated, attr_shape)
