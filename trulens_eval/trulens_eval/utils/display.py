import pandas as pd
from trulens_eval.schema.record import Record

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
