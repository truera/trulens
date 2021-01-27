import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import TestCase, main

import tensorflow as tf

from tests.unit.doi_test_base import DoiTestBase

assert(not tf.executing_eagerly())
class DoiTest(DoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
