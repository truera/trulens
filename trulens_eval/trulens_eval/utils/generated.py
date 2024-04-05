"""
Utilities for dealing with LLM-generated text.
"""

import logging
import re
from typing import Optional

from trulens_eval.utils.text import retab

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Error parsing LLM-generated text."""

    def __init__(
        self, expected: str, text: str, pattern: Optional[re.Pattern] = None
    ):
        super().__init__(
            f"Tried to find {expected}" +
            (f" using pattern {pattern.pattern}" if pattern else "") + " in\n" +
            retab(tab='  ', s=text)
        )
        self.text = text
        self.pattern = pattern


def validate_rating(rating) -> int:
    """Validate a rating is between 0 and 10."""

    if not 0 <= rating <= 10:
        raise ValueError('Rating must be between 0 and 10')

    return rating


# Various old patterns that didn't work as well:
# PATTERN_0_10: re.Pattern = re.compile(r"\s*([0-9]+)\s*$")
# PATTERN_0_10: re.Pattern = re.compile(r"\b([0-9]|10)(?=\D*$|\s*\.)")
PATTERN_0_10: re.Pattern = re.compile(r"([0-9]+)(?=\D*$)")
"""Regex that matches the last integer."""

PATTERN_NUMBER: re.Pattern = re.compile(r"([+-]?[0-9]+\.[0-9]*|[1-9][0-9]*|0)")
"""Regex that matches floating point and integer numbers."""

PATTERN_INTEGER: re.Pattern = re.compile(r"([+-]?[1-9][0-9]*|0)")
"""Regex that matches integers."""


def re_0_10_rating(s: str) -> int:
    """Extract a 0-10 rating from a string.
    
    If the string does not match an integer or matches an integer outside the
    0-10 range, raises an error instead. If multiple numbers are found within
    the expected 0-10 range, the smallest is returned.

    Args:
        s: String to extract rating from.

    Returns:
        int: Extracted rating. 
    
    Raises:
        ParseError: If no integers between 0 and 10 are found in the string.
    """

    matches = PATTERN_INTEGER.findall(s)
    if not matches:
        raise ParseError("int or float number", s, pattern=PATTERN_INTEGER)

    vals = set()
    for match in matches:
        try:
            vals.add(validate_rating(int(match)))
        except ValueError:
            pass

    if not vals:
        raise ParseError("0-10 rating", s)

    if len(vals) > 1:
        logger.warning(
            "Multiple valid rating values found in the string: %s", s
        )

    # Min to handle cases like "The rating is 8 out of 10."
    return min(vals)
