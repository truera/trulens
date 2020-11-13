import os
os.environ['NETLENS_BACKEND'] = 'pytorch'

from unittest import TestCase, main

from tests.unit.doi_test_base import DoiTestBase


class DoiTest(DoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
