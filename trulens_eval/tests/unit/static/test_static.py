"""
Static tests, i.e. ones that don't run anything substatial. This should find
issues that occur from merely importing trulens.
"""

from pprint import PrettyPrinter
from unittest import main
from unittest import TestCase


pp = PrettyPrinter()


class TestLens(TestCase):

    def setUp(self):
        pass

    def test_import_trulens_eval(self):
        import trulens_eval


if __name__ == '__main__':
    main()
