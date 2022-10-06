import os

os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import main
from unittest import TestCase

from tests.unit.backend_test_base import BackendTestBase


class BackendTest(BackendTestBase, TestCase):

    def setUp(self):
        super(BackendTest, self).setUp()


if __name__ == '__main__':
    main()
