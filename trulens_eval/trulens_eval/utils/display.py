import pandas as pd

def get_feedback_result(tru_record, feedback_name):
    feedback_calls = None
    for feedback_definition, future_result in tru_record.feedback_and_future_results:
        if feedback_definition.name == feedback_name:
            feedback_calls = future_result.result()
            break

    if feedback_calls is None:
        return pd.DataFrame()

    feedback_result = []
    for i in range(len(feedback_calls.calls)):
        args = feedback_calls.calls[i].model_dump()['args']
        ret = feedback_calls.calls[i].model_dump()['ret']
        feedback_result.append({**args, 'ret': ret})
    return pd.DataFrame(feedback_result)