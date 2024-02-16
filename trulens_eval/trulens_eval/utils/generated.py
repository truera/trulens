"""
Utilities for dealing with LLM-generated text.
"""

import logging
import re

from pydantic import BaseModel
from pydantic import field_validator
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class Rating(BaseModel):
    rating: int

    @field_validator('rating')
    def check_rating(cls, v):
        if not (0 <= v <= 10):
            raise ValueError('Rating must be between 0 and 10')
        return v


PATTERN_0_10: re.Pattern = re.compile(r"\s*([0-9]+)\s*$")
"""Regex for extracting a 0-10 rating.

We are assuming the score will always be the last part of the generated text
from LLM - hence we are matching for the last group of digits in the string.
"""


def re_0_10_rating(str_val: str) -> int:
    """Extract 0-10 rating from a string.
    
    If the string does not match, returns -10 instead."""

    matches = PATTERN_0_10.fullmatch(str_val)
    if not matches:
        # Try soft match
        matches = re.search(r'([0-9]+)(?=\D*$)', str_val)
        if not matches:
            logger.warning(f"0-10 rating regex failed to match on: '{str_val}'")
            return -10  # so this will be reported as -1 after division by 10

    try:
        rating = Rating(rating=int(matches.group()))
        return rating.rating
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return -10  # TODO: could consider incorporating re-asking and self-critique here with Instructor https://github.com/jxnl/instructor
