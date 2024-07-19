from enum import Enum


class DatabaseVersionException(Exception):
    """Exceptions for database version problems."""

    class Reason(Enum):
        """Reason for the version exception."""

        AHEAD = 1
        """Initialized database is ahead of the stored version."""

        BEHIND = 2
        """Initialized database is behind the stored version."""

        RECONFIGURED = 3
        """Initialized database differs in configuration compared to the stored
        version.
        
        Configuration differences recognized:
            - table_prefix
        
        """

    def __init__(self, msg: str, reason: Reason, **kwargs):
        self.reason = reason
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(msg)

    @classmethod
    def ahead(cls):
        """Create an ahead variant of this exception."""

        return cls(
            "Database schema is ahead of the expected revision. "
            "Please update to a later release of `trulens_eval`.",
            cls.Reason.AHEAD
        )

    @classmethod
    def behind(cls):
        """Create a behind variant of this exception."""

        return cls(
            "Database schema is behind the expected revision. "
            "Please upgrade it by running `tru.migrate_database()` "
            "or reset it by running `tru.reset_database()`.", cls.Reason.BEHIND
        )

    @classmethod
    def reconfigured(cls, prior_prefix: str):
        """Create a reconfigured variant of this exception.
        
        The only present reconfiguration that is recognized is a table_prefix
        change. A guess as to the prior prefix is included in the exception and
        message.
        """
        return cls(
            "Database has been reconfigured. "
            f"Please update it by running `tru.migrate_database(prior_prefix=\"{prior_prefix}\")`"
            " or reset it by running `tru.reset_database()`.",
            cls.Reason.RECONFIGURED,
            prior_prefix=prior_prefix
        )
