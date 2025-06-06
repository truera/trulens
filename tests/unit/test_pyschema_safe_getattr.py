from functools import cached_property
import sys
import unittest

from trulens.core.utils import constants as constant_utils
from trulens.core.utils.pyschema import safe_getattr


class SafeGetAttrTests(unittest.TestCase):
    def test_plain_and_missing_attribute(self):
        class X:
            class_attr = 123

            def __init__(self):
                self.inst_attr = 456

        x = X()
        # existing instance and class attrs
        self.assertEqual(safe_getattr(x, "inst_attr"), 456)
        self.assertEqual(safe_getattr(x, "class_attr"), 123)
        # missing attr raises AttributeError
        with self.assertRaises(AttributeError):
            safe_getattr(x, "no_such_attr")

    def test_property_invocation_and_error(self):
        class X:
            @property
            def good(self):
                return "value"

            @property
            def bad(self):
                raise RuntimeError("oops")

        x = X()
        # normal property invocation
        self.assertEqual(safe_getattr(x, "good"), "value")
        # get_prop=False should prevent invocation
        with self.assertRaises(ValueError):
            safe_getattr(x, "good", get_prop=False)
        # error in property should return an ERROR dict
        result = safe_getattr(x, "bad")
        self.assertIsInstance(result, dict)
        self.assertIn(constant_utils.ERROR, result)

    @unittest.skipUnless(
        sys.version_info >= (3, 12),
        "test_stdlib_cached_property requires Python 3.12+",
    )
    def test_stdlib_cached_property(self):
        class X:
            calls = 0

            @cached_property
            def cp(self):
                X.calls += 1
                return "cached"

        x = X()
        # first access computes once
        self.assertEqual(safe_getattr(x, "cp"), "cached")
        self.assertEqual(X.calls, 1)
        # second access uses the cache
        self.assertEqual(safe_getattr(x, "cp"), "cached")
        self.assertEqual(X.calls, 1)
