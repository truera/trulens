from enum import Enum


# snowflake object type
class OBJECTTYPE(str, Enum):
    EXTERNAL_AGENT = "EXTERNAL_AGENT"
    # CORTEX_SEARCH_SERVICE = "CORTEX_SEARCH_SERVICE"  # TODO: to be finalized
