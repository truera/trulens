# Cross-page constants to import from here to avoid circular imports
import os

LEADERBOARD_PAGE_NAME = "Leaderboard"
RECORDS_PAGE_NAME = "Records"
COMPARE_PAGE_NAME = "Compare"

RECORDS_LIMIT = 1000

CACHE_TTL = 60 * 15  # 15 minutes

PINNED_COL_NAME = "trulens.dashboard.pinned"
EXTERNAL_APP_COL_NAME = "trulens.dashboard.external_app"
HIDE_RECORD_COL_NAME = "trulens.dashboard.hide_record"

# NOTE: This is a flag to enable compatibility with Streamlit in Snowflake (SiS).
# SiS runs on Python 3.8, Streamlit 1.35.0, and does not support bidirectional custom components.
# As a result, enabling this flag will replace custom components in the dashboard with native Streamlit components.
SIS_COMPAT_FLAG = os.environ.get("SIS_COMPATIBILITY_MODE", None) == "true"
