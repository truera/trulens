import os
os.environ['NETLENS_BACKEND'] = 'tensorflow'

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from unittest import main, TestCase

from tests.tf2.unit.model_wrapper_test import ModelWrapperTest


class ModelWrapperTestNonEager(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = ModelWrapperTest(*args, **kwargs)


if __name__ == '__main__':
    main()
