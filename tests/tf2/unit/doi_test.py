import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import TestCase, main

from tests.unit.doi_test_base import DoiTestBase


class DoiTest(DoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
