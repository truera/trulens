import time

import pandas as pd
from trulens.core.schema.feedback import FeedbackDefinition
from trulens.core.schema.record import Record
from trulens.dashboard.ux.styles import CATEGORY


def get_icon(fdef: FeedbackDefinition, result: float) -> str:
    """
    Get the icon for a given feedback definition and result.

    Args:

    fdef : FeedbackDefinition
        The feedback definition
    result : float
        The result of the feedback

    Returns:
        str: The icon for the feedback
    """
    cat = CATEGORY.of_score(
        result or 0,
        higher_is_better=fdef.higher_is_better
        if fdef.higher_is_better is not None else True
    )
    return cat.icon
