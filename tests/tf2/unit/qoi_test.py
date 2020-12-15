import os
os.environ['TRULENS_BACKEND'] = 'tensorflow'

from unittest import TestCase, main

from tests.unit.qoi_test_base import QoiTestBase


class QoiTest(QoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
