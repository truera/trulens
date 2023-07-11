import numpy as np
from trulens.nn.backend import get_backend
from trulens.nn.quantities import LambdaQoI
from trulens.nn.quantities import MaxClassQoI


class DeterminismTestBase(object):

    def setUp(self):
        self.B = get_backend()
        # Create an example tensor to use for the tests.
        self.x = self.B.as_tensor(np.arange(10.0))

    def test_fprop(self):
        """fprop determinism test."""

        # Put the wrapped model into training mode which could cause
        # non-determinism if we don't handle it properly.
        # TODO: similar for other backends if appropriate
        self.model_nondet._model.train()

        out1 = self.model_nondet.fprop(model_args=[self.x])
        out2 = self.model_nondet.fprop(model_args=[self.x])

        self.assertTrue(np.allclose(out1, out2))

    def test_qprop(self):
        """qprop determinism test."""

        qoi = LambdaQoI(lambda x: x)

        # Put the wrapped model into training mode which could cause
        # non-determinism if we don't handle it properly.
        # TODO: similar for other backends if appropriate
        self.model_nondet._model.train()

        B = get_backend()

        grad1 = B.as_array(
            self.model_nondet.qoi_bprop(qoi=qoi, model_args=[self.x])
        )
        grad2 = B.as_array(
            self.model_nondet.qoi_bprop(qoi=qoi, model_args=[self.x])
        )

        self.assertTrue(np.allclose(grad1, grad2))
