import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import main
from unittest import TestCase

import tensorflow as tf
from tests.tf2.unit.model_wrapper_test import ModelWrapperTest

assert (not tf.executing_eagerly())
if __name__ == '__main__':
    main()
