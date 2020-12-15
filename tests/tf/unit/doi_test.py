import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import TestCase, main

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from tests.unit.doi_test_base import DoiTestBase


class DoiTest(DoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
