"""Tests for GetIndex step and Lens negative indexing."""

from unittest import TestCase

from trulens.core.utils.serial import GetIndex
from trulens.core.utils.serial import Lens


class TestGetIndex(TestCase):
    """Verify that GetIndex supports both positive and negative indices."""

    def test_get_index_positive(self):
        g = GetIndex(index=1)
        self.assertEqual(list(g.get(["a", "b", "c"])), ["b"])

        g_oob = GetIndex(index=5)
        with self.assertRaises(IndexError) as ctx:
            list(g_oob.get(["a", "b"]))
        self.assertIn("Index out of bounds: 5", str(ctx.exception))

    def test_get_index_negative(self):
        g = GetIndex(index=-1)
        self.assertEqual(list(g.get(["a", "b", "c"])), ["c"])

        g_first = GetIndex(index=-3)
        self.assertEqual(list(g_first.get(["a", "b", "c"])), ["a"])

        # Negative index on empty list should raise cleanly with bounds message
        with self.assertRaises(IndexError) as ctx:
            list(g.get([]))
        self.assertIn("Index out of bounds: -1", str(ctx.exception))

        # Out-of-bounds negative index
        g_oob = GetIndex(index=-4)
        with self.assertRaises(IndexError) as ctx:
            list(g_oob.get(["a", "b", "c"]))
        self.assertIn("Index out of bounds: -4", str(ctx.exception))

    def test_set_index_positive(self):
        g = GetIndex(index=0)
        self.assertEqual(g.set(["a", "b"], "x"), ["x", "b"])

        # Positive index pads with None
        g_expand = GetIndex(index=3)
        self.assertEqual(g_expand.set(["a"], "x"), ["a", None, None, "x"])

    def test_set_index_negative(self):
        g = GetIndex(index=-1)
        self.assertEqual(g.set(["a", "b"], "x"), ["a", "x"])

        # Negative index on empty list
        self.assertEqual(g.set([], "x"), ["x"])

        # Negative index on None initializes list
        self.assertEqual(g.set(None, "x"), ["x"])

        # Negative index prepends None when out of bounds
        g_prepend = GetIndex(index=-3)
        self.assertEqual(g_prepend.set(["b"], "x"), ["x", None, "b"])

    def test_lens_negative_indexing(self):
        lens = Lens().items[-1]
        data = {"items": [10, 20, 30]}
        self.assertEqual(lens.get_sole_item(data), 30)

        # Setting on empty list through Lens
        updated = lens.set({"items": []}, 99)
        self.assertEqual(updated["items"], [99])

        # Setting on missing path through Lens
        created = lens.set({}, 42)
        self.assertEqual(created["items"], [42])
