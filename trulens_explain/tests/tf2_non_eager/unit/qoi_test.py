import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import main
from unittest import TestCase

import tensorflow as tf
from tests.unit.qoi_test_base import QoiTestBase

assert (not tf.executing_eagerly())


class QoiTest(QoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
