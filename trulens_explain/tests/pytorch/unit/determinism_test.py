import os

from trulens.nn.models import get_model_wrapper

from tests.unit.determinism_test_base import DeterminismTestBase

os.environ["TRULENS_BACKEND"] = "pytorch"

from unittest import TestCase
from unittest import main

import torch
from torch.nn import Module


class DeterminismTest(DeterminismTestBase, TestCase):
    def setUp(self):
        super().setUp()

        class NonDet(Module):
            def __init__(this):
                super().__init__()
                # other non-deterministic layers?
                this.dropout1 = torch.nn.Dropout()
                this.dropout2 = torch.nn.Dropout()

            def forward(this, x):
                return this.dropout1(x) * this.dropout2(x)

        self.model_nondet = get_model_wrapper(NonDet())


if __name__ == "__main__":
    main()
