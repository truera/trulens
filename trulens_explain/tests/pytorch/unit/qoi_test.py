import os

os.environ['TRULENS_BACKEND'] = 'pytorch'

from unittest import main
from unittest import TestCase

from tests.unit.qoi_test_base import QoiTestBase


class QoiTest(QoiTestBase, TestCase):
    pass


if __name__ == '__main__':
    main()
