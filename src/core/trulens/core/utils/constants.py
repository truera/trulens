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

# Marks a score that no judge produced. Defined here so consumers of metric
# results do not need the optional trulens-feedback package.
UNPARSABLE_SCORE = -1.0
"""Sentinel score for "no score could be produced".

`generate_score` and `generate_score_and_reasons` raise
`trulens.feedback.generated.ParseError` rather than return this value. It still
appears where a metric records a failure as a score instead of raising: the
groundedness methods give it to a statement whose judge reply could not be
parsed (and report it when no statement was graded), the role adherence metric
returns it when it cannot read the judge's verdicts, aggregates return it when
every input was unparsable, and results stored by earlier versions may carry it.

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
