"""Regression test for _otel_id byte order and Python 3.10 compatibility.

int.from_bytes() requires byteorder on Python 3.10 (it only became optional,
defaulting to "big", in 3.11). _otel_id() omitted it, so every id derivation,
and therefore TraceAssembler.assemble(), raised TypeError on 3.10. The explicit
byteorder must be "big" so ids are unchanged on 3.11+.
"""

import hashlib
import unittest

from trulens.core.otel.client_hooks.tracing import _otel_id


class TestOtelIdByteOrder(unittest.TestCase):
    def test_matches_big_endian_interpretation(self):
        seed, bits = "trace:abc", 128
        size = bits // 8
        expected = int.from_bytes(
            hashlib.sha256(seed.encode()).digest()[:size], byteorder="big"
        )
        self.assertEqual(_otel_id(seed, bits), expected)

    def test_returns_int_without_raising(self):
        # Would raise TypeError on Python 3.10 before the fix.
        self.assertIsInstance(_otel_id("span:xyz", 64), int)
        self.assertEqual(_otel_id("span:xyz", 64), 15422170499572505575)

    def test_never_zero(self):
        self.assertGreaterEqual(_otel_id("anything", 64), 1)


if __name__ == "__main__":
    unittest.main()
