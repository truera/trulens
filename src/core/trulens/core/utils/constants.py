"""
This module contains common constants used throughout the trulens
"""

# Field/key name used to indicate a circular reference in dictified objects.
CIRCLE = "__tru_circular_reference"

# Field/key name used to indicate an exception in property retrieval (properties
# execute code in property.fget).
ERROR = "__tru_property_error"

# Key for indicating non-serialized objects in json dumps.
NOSERIO = "__tru_non_serialized_object"

# Key of structure where class information is stored.
CLASS_INFO = "tru_class_info"

ALL_SPECIAL_KEYS = set([CIRCLE, ERROR, CLASS_INFO, NOSERIO])

# Judge-backed feedback methods return this when the judge's reply could not
# be parsed into a score. Defined here so consumers of metric results do not
# need the optional trulens-feedback package.
UNPARSABLE_SCORE = -1.0
"""Sentinel score returned when a judge's reply could not be parsed.

Feedback scores are otherwise normalized to the 0 to 1 range, so this value
sits outside it on purpose and is never a verdict. TruLens's own aggregates
skip it and guardrails reject it. Integrators that compare a score against a
threshold should check [is_unparsable_score][] first.
"""


def is_unparsable_score(score: float) -> bool:
    """Whether `score` is the sentinel for an unparsable judge reply.

    Only the exact sentinel counts. NaN and infinities are not the sentinel
    and are left for the caller to handle.
    """
    return score == UNPARSABLE_SCORE
