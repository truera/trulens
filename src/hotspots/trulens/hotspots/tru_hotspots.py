from typing import Optional

import pandas as pd
from trulens.core.schema import types as types_schema
from trulens.core.session import TruSession
from trulens.hotspots import HotspotsConfig
from trulens.hotspots import hotspots_as_df


def get_hotspots(
    session: TruSession,
    app_ids: Optional[list[types_schema.AppID]] = None,
    hotspots_config: Optional[HotspotsConfig] = None,
    feedback: Optional[str] = None,
) -> pd.DataFrame:
    """Get hotspots for a TruLens session.

    Args:
        session: TruLens session

        app_ids: A list of app IDs to filter records by. If empty or not given, all
              apps' records will be returned.

        hotspots_config: A hotspots configuration. A default one based on feedbacks will be used
              if none given.

        feedback: The name of the feedback to be used. The first feedback will be used if none given

    Returns:
        Data frame with hotspots
    """
    df, feedback_names = session.get_records_and_feedback(app_ids=app_ids)

    if hotspots_config is None:
        if feedback is None:
            feedback = feedback_names[0]

        hotspots_config = HotspotsConfig(score_column=feedback)

    columns_to_be_skipped = (
        ["record_json"]
        + [f for f in feedback_names if f != hotspots_config.score_column]
        + [f + "_calls" for f in feedback_names]
    )

    hotspots_config.skip_columns += columns_to_be_skipped

    return hotspots_as_df(hotspots_config, df)
