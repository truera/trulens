from enum import Enum


class Mode(str, Enum):
    LOG_INGESTION = "LOG_INGESTION"
    APP_INVOCATION = "APP_INVOCATION"

    @classmethod
    def is_valid_mode(cls, key) -> bool:
        return key in cls.__members__.values()
