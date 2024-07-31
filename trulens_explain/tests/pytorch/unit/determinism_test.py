import os

from tests.unit.determinism_test_base import DeterminismTestBase
from trulens.nn.models import get_model_wrapper

os.environ['TRULENS_BACKEND'] = 'pytorch'

from unittest import main
from unittest import TestCase

import torch
from torch.nn import Module


class DeterminismTest(DeterminismTestBase, TestCase):

    def setUp(self):
        super(DeterminismTest, self).setUp()

        class NonDet(Module):

            def __init__(this):
                super(NonDet, this).__init__()
                # other non-deterministic layers?
                this.dropout1 = torch.nn.Dropout()
                this.dropout2 = torch.nn.Dropout()

            def forward(this, x):
                return this.dropout1(x) * this.dropout2(x)

        self.model_nondet = get_model_wrapper(NonDet())


if __name__ == '__main__':
    main()
