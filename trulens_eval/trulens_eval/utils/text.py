"""
Utilities for user-facing text generation.
"""

import logging
import sys

from trulens_eval.utils.python import safe_hasattr

logger = logging.getLogger(__name__)

if safe_hasattr(sys.stdout, "reconfigure"):
    # Some stdout can't handle the below emojis (like terminal). This will skip over the emoji printing
    sys.stdout.reconfigure(errors="replace")

UNICODE_STOP = "🛑"
UNICODE_CHECK = "✅"
UNICODE_YIELD = "⚡"
UNICODE_HOURGLASS = "⏳"
UNICODE_CLOCK = "⏰"
UNICODE_SQUID = "🦑"
UNICODE_LOCK = "🔒"


def make_retab(tab):

    def retab(s):
        lines = s.split("\n")
        return tab + f"\n{tab}".join(lines)

    return retab


def retab(s: str, tab: str = "\t"):
    return make_retab(tab)(s)
