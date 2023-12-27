"""
Utilities for dealing with LLM-generated text.
"""

import logging
import re

logger = logging.getLogger(__name__)

# for extracting the 0-10 rating, we are assuming the score will
# always be the last part of the generated text from LLM - hence we are matching for the last
# group of digits in the string
pat_0_10 = re.compile(r"\s*([0-9]+)\s*$")


def re_0_10_rating(str_val):
    matches = pat_0_10.fullmatch(str_val)
    if not matches:
        # Try soft match
        matches = re.search(r'([0-9]+)(?=\D*$)', str_val)
        if not matches:
            logger.warning(f"0-10 rating regex failed to match on: '{str_val}'")
            return -10  # so this will be reported as -1 after division by 10

    return int(matches.group())