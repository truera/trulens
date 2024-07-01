import pandas as pd
from trulens_eval.schema.record import Record
from trulens_eval.schema.feedback import FeedbackDefinition
from trulens_eval.ux.styles import CATEGORY


def get_feedback_result(tru_record: Record, feedback_name: str) -> pd.DataFrame:
    feedback_calls = next(
        (
            future_result.result()
            for feedback_definition, future_result in
            tru_record.feedback_and_future_results
            if feedback_definition.name == feedback_name
        ), None
    )
    if feedback_calls is None:
        return pd.DataFrame()
    feedback_result = [
        {
            **call.model_dump()['args'], 'ret': call.model_dump()['ret']
        } for call in feedback_calls.calls
    ]
    return pd.DataFrame(feedback_result)


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
