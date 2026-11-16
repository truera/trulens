"""Tests for LensedDict."""

from unittest import TestCase

from trulens.core.utils.serial import Lens
from trulens.core.utils.serial import LensedDict


class TestLensedDict(TestCase):
    """Verify that LensedDict supports Lens lookups via contains, get, and item access."""

    def test_lensed_dict_contains_with_lens(self):
        d = LensedDict({"a": {"b": 123, "c": None}})
        self.assertTrue(Lens().a.b in d)
        self.assertTrue(Lens().a.c in d)
        self.assertFalse(Lens().a.d in d)
        self.assertFalse(Lens().x.y in d)

    def test_lensed_dict_contains_with_regular_key(self):
        d = LensedDict({"a": 1, "b": 2})
        self.assertTrue("a" in d)
        self.assertFalse("c" in d)

    def test_lensed_dict_get_with_lens(self):
        d = LensedDict({"a": {"b": 123, "c": None}})
        self.assertEqual(d.get(Lens().a.b), 123)
        self.assertIsNone(d.get(Lens().a.c))
        self.assertIsNone(d.get(Lens().a.d))
        self.assertEqual(d.get(Lens().a.d, "default"), "default")
        self.assertEqual(d.get(Lens().x.y, 42), 42)

    def test_lensed_dict_get_with_regular_key(self):
        d = LensedDict({"a": 1})
        self.assertEqual(d.get("a"), 1)
        self.assertIsNone(d.get("b"))
        self.assertEqual(d.get("b", 99), 99)

    def test_lensed_dict_getitem_and_setitem(self):
        d = LensedDict({"a": {"b": 10}})
        self.assertEqual(d[Lens().a.b], 10)
        d[Lens().a.b] = 20
        self.assertEqual(d[Lens().a.b], 20)
        self.assertTrue(Lens().a.b in d)

    def test_lensed_dict_ambiguous_lens(self):
        d = LensedDict({"items": [1, 2]})
        ambiguous = Lens()["items"][:]
        # When a lens matches multiple items:
        # contains returns False and get returns default.
        self.assertFalse(ambiguous in d)
        self.assertEqual(d.get(ambiguous, "fallback"), "fallback")
        self.assertIsNone(d.get(ambiguous))
        # Under normal (non-optimized) execution, direct indexing raises AssertionError from Lens.get_sole_item.
        if __debug__:
            with self.assertRaises(AssertionError):
                _ = d[ambiguous]
