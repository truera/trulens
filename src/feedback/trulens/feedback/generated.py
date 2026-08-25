"""Utilities for dealing with LLM-generated text."""

import logging
import re
from typing import Optional
import warnings

from trulens.core.utils import text as text_utils

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Error parsing LLM-generated text."""

    def __init__(
        self, expected: str, text: str, pattern: Optional[re.Pattern] = None
    ):
        super().__init__(
            f"Tried to find {expected}"
            + (f" using pattern {pattern.pattern}" if pattern else "")
            + " in\n"
            + text_utils.retab(tab="  ", s=text)
        )
        self.text = text
        self.pattern = pattern


# Various old patterns that didn't work as well:
# PATTERN_0_10: re.Pattern = re.compile(r"\s*([0-9]+)\s*$")
# PATTERN_0_10: re.Pattern = re.compile(r"\b([0-9]|10)(?=\D*$|\s*\.)")
PATTERN_0_10: re.Pattern = re.compile(r"([0-9]+)(?=\D*$)")
"""Regex that matches the last integer."""

PATTERN_NUMBER: re.Pattern = re.compile(r"([+-]?[0-9]+\.[0-9]*|[1-9][0-9]*|0)")
"""Regex that matches floating point and integer numbers."""

PATTERN_INTEGER: re.Pattern = re.compile(r"([+-]?[1-9][0-9]*|0)")
"""Regex that matches integers."""


def _ratings_in_range(
    s: str, min_score_val: int, max_score_val: int, allow_decimal: bool
) -> set:
    """Every number in ``s`` that falls inside the rating scale.

    Args:
        s: String to read numbers from.

        min_score_val: Minimum value of the rating scale.

        max_score_val: Maximum value of the rating scale.

        allow_decimal: Whether to capture decimal numbers rather than truncate.

    Returns:
        set: The in-range values. Out-of-range numbers are logged and dropped.
    """
    found = set()
    for match in PATTERN_NUMBER.findall(s):
        rating = float(match) if allow_decimal else int(float(match))
        if min_score_val <= rating <= max_score_val:
            found.add(rating)
        else:
            logger.warning(
                "Rating must be in [%s, %s].", min_score_val, max_score_val
            )
    return found


def _strip_scale_statement(
    s: str, min_score_val: int, max_score_val: int
) -> str:
    """Remove an explicit statement of the rating scale from a judge's answer.

    Covers the two shapes judges actually produce: a range ("0 to 3", "0-3",
    "between 0 and 3") and a legend ("0 = irrelevant, 3 = highly relevant").
    Only the configured bounds are removed, so a number that happens to equal a
    bound but is the actual rating is left alone unless it appears inside one of
    those shapes.

    Args:
        s: String to strip.

        min_score_val: Minimum value of the rating scale.

        max_score_val: Maximum value of the rating scale.

    Returns:
        str: The string with any scale statement removed.
    """
    lo = re.escape(str(min_score_val))
    hi = re.escape(str(max_score_val))
    patterns = (
        rf"\b{lo}\s*(?:to|through|and|up to|\.\.\.?|[-‒-―])\s*{hi}\b",
        rf"\b{lo}\s*=",
        rf"\b{hi}\s*=",
    )
    for pattern in patterns:
        s = re.sub(pattern, " ", s, flags=re.IGNORECASE)
    return s


def re_configured_rating(
    s: str,
    min_score_val: int = 0,
    max_score_val: int = 3,
    allow_decimal: bool = False,
) -> int:
    """Extract a {min_score_val}-{max_score_val} rating from a string. Configurable
    to the ranges like 4-point Likert scale or binary (0 or 1).

    If the string does not match an integer/a float or matches an integer/a
    float outside the {min_score_val} - {max_score_val} range, raises an error
    instead. If multiple numbers are found within the expected 0-10 range, the
    smallest is returned.

    Args:
        s: String to extract rating from.

        min_score_val: Minimum value of the rating scale.

        max_score_val: Maximum value of the rating scale.

        allow_decimal: Whether to allow and capture decimal numbers (floats).

    Returns:
        int: Extracted rating.

    Raises:
        ParseError: If no integers/floats between 0 and 10 are found in the string.
    """
    if max_score_val <= min_score_val:
        raise ValueError("Max score must be greater than min score.")

    if not PATTERN_NUMBER.search(s):
        raise ParseError("int or float number", s, pattern=PATTERN_NUMBER)

    # A judge that repeats its own scale ("on a scale of 0 to 3, I rate this 3") is
    # naming the bounds, not scoring. Those bounds are in range, so without this they
    # become candidate ratings and the min() below returns the floor instead of the
    # rating. Drop an explicit scale statement and read the rest, so the bounds cannot
    # be re-injected further down. "N out of M" contains no scale statement, so it is
    # untouched and keeps its existing behaviour.
    without_scale = _strip_scale_statement(s, min_score_val, max_score_val)
    considered = without_scale if without_scale != s else s

    vals = _ratings_in_range(
        considered, min_score_val, max_score_val, allow_decimal
    )
    if not vals and considered is not s:
        # Stripping took the only numbers with it, so fall back to the whole string
        # rather than raising on text the old behaviour could read.
        considered = s
        vals = _ratings_in_range(
            considered, min_score_val, max_score_val, allow_decimal
        )

    if not vals:
        raise ParseError(f"{min_score_val}-{max_score_val} rating", s)

    if len(vals) > 1:
        logger.warning(
            "Multiple valid rating values found in the string: %s", s
        )

    # Min to handle cases like "The rating is 1 out of 3."
    return min(vals)


def re_0_10_rating(s: str) -> int:
    """Extract a 0-10 rating from a string.

    If the string does not match an integer/a float or matches an integer/a
    float outside the 0-10 range, raises an error instead. If multiple numbers
    are found within the expected 0-10 range, the smallest is returned.

    Args:
        s: String to extract rating from.

    Returns:
        int: Extracted rating.

    Raises:
        ParseError: If no integers/floats between 0 and 10 are found in the
            string.
    """

    return re_configured_rating(
        s, min_score_val=0, max_score_val=10, allow_decimal=True
    )


def validate_rating(rating: str) -> bool:
    warnings.warn(
        "This method is deprecated. "
        "Use try/catch with `trulens.feedback.generated.re_0_10_rating` instead:\n"
        "  ```python\n"
        "  try:\n"
        "      re_0_10_rating(rating)\n"
        "  except ParseError:\n"
        "      # not validated\n"
        "  # validated\n"
        "```",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        re_0_10_rating(rating)
    except ParseError:
        return False
    return True
