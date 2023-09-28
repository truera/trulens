"""
Utilities for dealing with LLM-generated text.
"""

import logging
import re

logger = logging.getLogger(__name__)

pat_1_10 = re.compile(r"\s*([1-9][0-9]*)\s*")


def re_1_10_rating(str_val):
    matches = pat_1_10.fullmatch(str_val)
    if not matches:
        # Try soft match
        matches = re.search('[1-9][0-9]*', str_val)
        if not matches:
            logger.warning(f"1-10 rating regex failed to match on: '{str_val}'")
            return -10  # so this will be reported as -1 after division by 10

    return int(matches.group())