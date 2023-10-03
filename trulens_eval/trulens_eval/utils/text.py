"""
Utilities for user-facing text generation.
"""

import logging
import sys

logger = logging.getLogger(__name__)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


UNICODE_STOP = "🛑"
UNICODE_CHECK = "✅"
UNICODE_YIELD = "⚡"
UNICODE_HOURGLASS = "⏳"
UNICODE_CLOCK = "⏰"
UNICODE_SQUID = "🦑"
UNICODE_LOCK = "🔒"