import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import main, TestCase
from tests.tf2.unit.model_wrapper_test import ModelWrapperTest

import tensorflow as tf
assert (not tf.executing_eagerly())
if __name__ == '__main__':
    main()
