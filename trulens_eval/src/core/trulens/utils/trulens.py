"""
Utilities for app components provided as part of the trulens_eval package.
Currently organizes all such components as "Other".
"""

import time
from typing import Type

import pandas as pd
from trulens.core import app
from trulens.core.schema.record import Record
from trulens.utils.pyschema import Class


class Other(app.Other, app.TrulensComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Other]


def constructor_of_class(cls: Class) -> Type[app.TrulensComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls):
            return view

    raise TypeError(f'Unknown trulens component type with class {cls}')


def component_of_json(json: dict) -> app.TrulensComponent:
    cls = Class.of_class_info(json)

    view = constructor_of_class(cls)

    return view(json)


def get_feedback_result(
    tru_record: Record, feedback_name: str, timeout: int = 60
) -> pd.DataFrame:
    """
    Retrieve the feedback results for a given feedback name from a TruLens record.

    Args:
        tru_record (Record): The record containing feedback and future results.
        feedback_name (str): The name of the feedback to retrieve results for.

    Returns:
        pd.DataFrame: A DataFrame containing the feedback results. If no feedback
                      results are found, an empty DataFrame is returned.
    """
    start_time = time.time()
    feedback_calls = None

    while time.time() - start_time < timeout:
        feedback_calls = next(
            (
                future_result.result()
                for feedback_definition, future_result in
                tru_record.feedback_and_future_results
                if feedback_definition.name == feedback_name
            ), None
        )
        if feedback_calls is not None:
            break
        time.sleep(1)  # Wait for 1 second before checking again

    if feedback_calls is None:
        raise TimeoutError(
            f"Feedback for '{feedback_name}' not available within {timeout} seconds."
        )

    # Ensure feedback_calls is iterable
    if not hasattr(feedback_calls, '__iter__'):
        raise ValueError('feedback_calls is not iterable')

    feedback_result = [
        {
            **call.model_dump()['args'], 'ret': call.model_dump()['ret']
        } for call in feedback_calls.calls
    ]
    return pd.DataFrame(feedback_result)
