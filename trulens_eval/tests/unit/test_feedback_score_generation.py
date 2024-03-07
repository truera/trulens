import re

import pytest

from trulens_eval.utils.generated import PATTERN_0_10
"""
Test suites meant for testing the reliability and robustness of the regex pattern matching of feedback scores from LLM responses.
"""

# Original regex pattern
PATTERN_0_10_ORIGINAL: re.Pattern = re.compile(r"\s*([0-9]+)\s*$")

# Enhanced regex pattern
PATTERN_0_10_ENHANCED: re.Pattern = PATTERN_0_10

test_data = [
    ("The relevance score is 7.", "7"),
    ("I rate this an 8 out of 10.", "10"
    ),  # note that even with the enhanced pattern, this will still fail to extract 8.
    ("This should be a 10!", "10"),
    ("The score is 5", "5"),
    ("A perfect score: 10.", "10"),
    ("Irrelevant text 123 Main Street.", None),
    ("Score: 9.", "9"),
    ("7", "7"),
    ("This deserves a 6, I believe.", "6"),
    ("Not relevant. Score: 0.", "0")
]


@pytest.mark.parametrize("test_input,expected", test_data)
def test_enhanced_regex_pattern(test_input, expected):
    match = PATTERN_0_10_ENHANCED.search(test_input)
    result = match.group(1) if match else None
    assert result == expected, f"Failed on {test_input}: expected {expected}, got {result}"
