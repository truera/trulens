"""
Tests for serial.py:Lens class.
"""


from pprint import PrettyPrinter
from unittest import main
from unittest import TestCase

from munch import Munch

from trulens_eval.utils.serial import GetAttribute, GetItem, Lens



pp = PrettyPrinter()

class TestLens(TestCase):

    def setUp(self):

        # Don't try to print this object.
        self.obj1 = dict(
            outerkey = Munch(
                intkey = 42,
                strkey = "hello",
                seqkey=[1,2,3,4,5],
                innerkey = "placeholder"
            ),
            outerstr = "lalala",
            outerint = 0xdeadbeef
        )

        # Because it contains itself:
        self.obj1['outerkey'].innerkey = self.obj1

    def testStepsGet(self):

        # GetItem, GetAttribute
        self.assertEqual(
            Lens(path=(GetItem(item="outerkey"),
                       GetAttribute(attribute="strkey"),
                       )).get_sole_item(self.obj1),
            "hello"
        )

        # GetItemOrAttribute
        self.assertEqual(
            Lens().outerkey.intkey.get_sole_item(self.obj1),
            42
        )

        # GetIndex
        self.assertEqual(
            Lens().outerkey.seqkey[2].get_sole_item(self.obj1),
            3
        )

        # GetSlice
        self.assertEqual(
            list(Lens().outerkey.seqkey[3:1:-1].get(self.obj1)),
            [4, 3]
        )

        # GetIndices
        self.assertEqual(
            list(Lens().outerkey.seqkey[1,3].get(self.obj1)),
            [2, 4]
        )

        # GetItems
        self.assertEqual(
            list(Lens()['outerstr', 'outerint'].get(self.obj1)),
            ["lalala", 0xdeadbeef]
        )


    def testStepsSet(self):

        # GetItem, GetAttribute
        obj1 = Lens(
            path=(GetItem(item="outerkey"),
                  GetAttribute(attribute="strkey"),
            )
        ).set(self.obj1, "not hello")

        self.assertEqual(
            obj1['outerkey'].strkey,
            "not hello"
        )

        # GetItemOrAttribute
        obj1 = Lens().outerkey.intkey.set(self.obj1, 43)
        self.assertEqual(
            obj1.outerkey.intkey,
            43
        )

        # GetIndex
        self.assertEqual(
            Lens().outerkey.seqkey[2].get_sole_item(self.obj1),
            3
        )

        # GetSlice
        self.assertEqual(
            list(Lens().outerkey.seqkey[3:1:-1].get(self.obj1)),
            [4, 3]
        )

        # GetIndices
        self.assertEqual(
            list(Lens().outerkey.seqkey[1,3].get(self.obj1)),
            [2, 4]
        )

        # GetItems
        self.assertEqual(
            list(Lens()['outerstr', 'outerint'].get(self.obj1)),
            ["lalala", 0xdeadbeef]
        )

if __name__ == '__main__':
    main()
