"""
Utilities for dealing with LLM-generated text.
"""

import logging
import re
from pydantic import BaseModel, field_validator, ValidationError


logger = logging.getLogger(__name__)


class Rating(BaseModel):
    rating: int

    @field_validator('rating')
    def check_rating(cls, v):
        if not (0 <= v <= 10):
            raise ValueError('Rating must be between 0 and 10')
        return v

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

    try:
        rating = Rating(rating=int(matches.group()))
        return rating.rating
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return -10  # TODO: could consider incorporating re-asking and self-critique here with Instructor https://github.com/jxnl/instructor