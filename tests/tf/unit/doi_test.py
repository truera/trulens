import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

import importlib
from unittest import main
from unittest import TestCase

import tensorflow as tf

if tf.__version__.startswith("1"):
    Graph = importlib.import_module("tensorflow.Graph")
    placeholder = importlib.import_module("tensorflow.placeholder")
    deprecation = importlib.import_module("tensorflow.python.util.deprecation")
    deprecation._PRINT_DEPRECATION_WARNINGS = False
else:
    raise RuntimeError(
        f"Running Tensorflow 1 tests with incorrect version of Tensorflow. Expected 1.x, got {tf.__version__}"
    )

from tests.unit.doi_test_base import DoiTestBase
from trulens.nn.models import get_model_wrapper


class DoiTest(DoiTestBase, TestCase):

    def setUp(self):
        super(DoiTest, self).setUp()

        graph = Graph()

        with graph.as_default():
            l0 = placeholder('float32', (None, 1))
            l1 = self.l1_coeff * (l0**self.l1_exp)
            l2 = self.l2_coeff * (l1**self.l2_exp)

        self.model = get_model_wrapper(
            graph,
            input_tensors=l0,
            output_tensors=l2,
            internal_tensor_dict=dict(layer0=l0, layer1=l1, layer2=l2)
        )

        self.layer0 = "input"
        self.layer1 = "layer1"
        self.layer2 = "layer2"


if __name__ == '__main__':
    main()
