from enum import Enum


# snowflake object type
class ObjectType(str, Enum):
    EXTERNAL_AGENT = "EXTERNAL AGENT"


class RunStatus(str, Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"
