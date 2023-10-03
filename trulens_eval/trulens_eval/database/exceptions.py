from enum import Enum


class DatabaseVersionException(Exception):

    class Reason(Enum):
        AHEAD = 1
        BEHIND = 2

    def __init__(self, msg: str, reason: Reason):
        self.reason = reason
        super().__init__(msg)

    @classmethod
    def ahead(cls):
        return cls(
            "Database schema is ahead of the expected revision. "
            "Please update to a later release of `trulens_eval`",
            cls.Reason.AHEAD
        )

    @classmethod
    def behind(cls):
        return cls(
            "Database schema is behind the expected revision. "
            "Please upgrade it by running `tru.migrate_database()` or reset it by running `tru.reset_database()`",
            cls.Reason.BEHIND
        )
