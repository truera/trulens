import os

os.environ['TRULENS_BACKEND'] = 'keras'

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from unittest import main
from unittest import TestCase

from tests.unit.qoi_test_base import QoiTestBase


class QoiTest(QoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
