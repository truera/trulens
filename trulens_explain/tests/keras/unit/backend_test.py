import os

os.environ["TRULENS_BACKEND"] = "keras"

from unittest import main
from unittest import TestCase

from tests.unit.backend_test_base import BackendTestBase


class BackendTest(BackendTestBase, TestCase):
    def setUp(self):
        super().setUp()


if __name__ == "__main__":
    main()
