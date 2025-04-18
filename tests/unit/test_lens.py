"""Tests for serial.py:Lens class."""

from pprint import PrettyPrinter
from unittest import TestCase

from munch import Munch
from trulens.core.utils import serial as serial_utils

pp = PrettyPrinter()


class TestLens(TestCase):
    def setUp(self):
        self.obj1 = dict(
            outerkey=Munch(
                intkey=42,
                strkey="hello",
                seqkey=[1, 2, 3, 4, 5],
                innerkey="placeholder",
            ),
            outerstr="lalala",
            outerint=0xDEADBEEF,
        )

    def testParse(self):
        # GetItemOrAttribute
        with self.subTest("GetItemOrAttribute"):
            self.assertEqual(
                serial_utils.Lens().of_string("outerkey.intkey"),
                serial_utils.Lens().outerkey.intkey,
            )

        # GetIndex
        with self.subTest("GetIndex"):
            self.assertEqual(
                serial_utils.Lens().of_string("outerkey.seqkey[2]"),
                serial_utils.Lens().outerkey.seqkey[2],
            )

        # GetSlice
        with self.subTest("GetSlice"):
            self.assertEqual(
                serial_utils.Lens().of_string("outerkey.seqkey[3:1:-1]"),
                serial_utils.Lens().outerkey.seqkey[3:1:-1],
            )

        # GetIndices
        with self.subTest("GetIndices"):
            self.assertEqual(
                serial_utils.Lens().of_string("outerkey.seqkey[1,3]"),
                serial_utils.Lens().outerkey.seqkey[1, 3],
            )

        # GetItems
        with self.subTest("GetItems"):
            self.assertEqual(
                serial_utils.Lens().of_string("['outerstr', 'outerint']"),
                serial_utils.Lens()["outerstr", "outerint"],
            )

        # Collect
        with self.subTest("Collect"):
            self.assertEqual(
                # note we are not manually collecting from the generator here, collect does it for us
                serial_utils.Lens().of_string(
                    "['outerstr', 'outerint'].collect()"
                ),
                serial_utils.Lens()["outerstr", "outerint"].collect(),
            )

    def testStepsGet(self):
        # GetItem, GetAttribute
        with self.subTest("GetItem,GetAttribute"):
            self.assertEqual(
                serial_utils.Lens(
                    path=(
                        serial_utils.GetItem(item="outerkey"),
                        serial_utils.GetAttribute(attribute="strkey"),
                    )
                ).get_sole_item(self.obj1),
                "hello",
            )

        # GetItemOrAttribute
        with self.subTest("GetItemOrAttribute"):
            self.assertEqual(
                serial_utils.Lens().outerkey.intkey.get_sole_item(self.obj1), 42
            )

        # GetIndex
        with self.subTest("GetIndex"):
            self.assertEqual(
                serial_utils.Lens().outerkey.seqkey[2].get_sole_item(self.obj1),
                3,
            )

        # GetSlice
        with self.subTest("GetSlice"):
            self.assertEqual(
                list(
                    serial_utils.Lens().outerkey.seqkey[3:1:-1].get(self.obj1)
                ),
                [4, 3],
            )

        # GetIndices
        with self.subTest("GetIndices"):
            self.assertEqual(
                list(serial_utils.Lens().outerkey.seqkey[1, 3].get(self.obj1)),
                [2, 4],
            )

        # GetItems
        with self.subTest("GetItems"):
            self.assertEqual(
                list(
                    serial_utils.Lens()["outerstr", "outerint"].get(self.obj1)
                ),
                ["lalala", 0xDEADBEEF],
            )

        # Collect
        with self.subTest("Collect"):
            self.assertEqual(
                # note we are not manually collecting from the generator here, collect does it for us
                serial_utils.Lens()["outerstr", "outerint"]
                .collect()
                .get_sole_item(self.obj1),
                ["lalala", 0xDEADBEEF],
            )

    def testStepsSet(self):
        # NOTE1: lens vs. python expression differences: Lens steps GetItems and
        # GetIndices do not have corresponding python list semantics. They do
        # with pandas dataframes and numpy arrays, respectively, though.

        # GetItem, GetAttribute
        with self.subTest("GetItem,GetAttribute"):
            self.assertEqual(self.obj1["outerkey"].strkey, "hello")
            obj1 = serial_utils.Lens(
                path=(
                    serial_utils.GetItem(item="outerkey"),
                    serial_utils.GetAttribute(attribute="strkey"),
                )
            ).set(self.obj1, "not hello")
            self.assertEqual(obj1["outerkey"].strkey, "not hello")

        # GetItemOrAttribute
        with self.subTest("GetItemOrAttribute"):
            self.assertEqual(self.obj1["outerkey"].intkey, 42)
            obj1 = serial_utils.Lens()["outerkey"].intkey.set(self.obj1, 43)
            self.assertEqual(obj1["outerkey"].intkey, 43)

        # GetIndex
        with self.subTest("GetIndex"):
            self.assertEqual(self.obj1["outerkey"].seqkey[2], 3)
            obj1 = serial_utils.Lens()["outerkey"].seqkey[2].set(self.obj1, 4)
            self.assertEqual(obj1["outerkey"].seqkey[2], 4)

        # Setting lenses that produce multiple things is not supported / does not work.

        # GetSlice
        with self.subTest("GetSlice"):
            self.assertEqual(self.obj1["outerkey"].seqkey[3:1:-1], [4, 3])
            obj1 = (
                serial_utils.Lens()["outerkey"]
                .seqkey[3:1:-1]
                .set(self.obj1, 43)
            )
            self.assertEqual(obj1["outerkey"].seqkey[3:1:-1], [43, 43])

        # GetIndices
        with self.subTest("GetIndices"):
            self.assertEqual(
                [
                    self.obj1["outerkey"].seqkey[1],
                    self.obj1["outerkey"].seqkey[3],
                ],  # NOTE1
                [2, 4],
            )
            obj1 = (
                serial_utils.Lens()["outerkey"].seqkey[1, 3].set(self.obj1, 24)
            )
            self.assertEqual(
                [
                    obj1["outerkey"].seqkey[1],
                    obj1["outerkey"].seqkey[3],
                ],  # NOTE1
                [24, 24],
            )

        # GetItems
        with self.subTest("GetItems"):
            self.assertEqual(
                [self.obj1["outerstr"], self.obj1["outerint"]],  # NOTE1
                ["lalala", 0xDEADBEEF],
            )
            obj1 = serial_utils.Lens()["outerstr", "outerint"].set(
                self.obj1, "still not hello 420"
            )
            self.assertEqual(
                [obj1["outerstr"], obj1["outerint"]],  # NOTE1
                ["still not hello 420", "still not hello 420"],
            )

        # Collect cannot be set.
