import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
assert(not tf.executing_eagerly())
from unittest import main, TestCase
from tests.tf2.unit.model_wrapper_test import ModelWrapperTest

if __name__ == '__main__':
    main()
