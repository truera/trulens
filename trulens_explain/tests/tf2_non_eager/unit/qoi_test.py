import os

os.environ["TRULENS_BACKEND"] = "tensorflow"

from unittest import TestCase
from unittest import main

import tensorflow as tf

from tests.unit.qoi_test_base import QoiTestBase

assert not tf.executing_eagerly()


class QoiTest(QoiTestBase, TestCase):
    pass


if __name__ == "__main__":
    main()
