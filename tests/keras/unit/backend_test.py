import os

os.environ['TRULENS_BACKEND'] = 'keras'

from unittest import main
from unittest import TestCase

from tests.unit.backend_test_base import BackendTestBase


class AxiomsTest(BackendTestBase, TestCase):

    def setUp(self):
        super(AxiomsTest, self).setUp()


if __name__ == '__main__':
    main()
