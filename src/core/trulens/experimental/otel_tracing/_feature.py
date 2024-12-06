"""Utilities for managing optional requirements of the experimental otel_tracing
feature."""

from sqlalchemy.orm import configure_mappers
from trulens.core import experimental
from trulens.core.utils import imports as import_utils

# from trulens.core import session as core_session # circular import


FEATURE = experimental.Feature.OTEL_TRACING
"""Feature controlling the use of this module."""

REQUIREMENT = import_utils.format_import_errors(
    ["opentelemetry-api", "opentelemetry-sdk", "trulens-otel-semconv"],
    purpose="otel_tracing experimental feature",
)
"""Optional modules required for the otel_tracing experimental feature."""

with import_utils.OptionalImports(REQUIREMENT) as oi:
    from opentelemetry import sdk
    from opentelemetry import trace
    from trulens.otel.semconv import trace as trulens_otel_semconv_trace


class _FeatureSetup(experimental._FeatureSetup):
    """Utilities for managing the otel_tracing experimental feature."""

    FEATURE = FEATURE
    REQUIREMENT = REQUIREMENT

    @staticmethod
    def assert_optionals_installed():
        """Asserts that the optional requirements for the otel_tracing feature are
        installed."""
        oi.assert_installed([sdk, trace, trulens_otel_semconv_trace])

    @staticmethod
    def are_optionals_installed():
        """Checks if the optional requirements for the otel_tracing feature are
        installed."""
        return not any(
            import_utils.is_dummy(m)
            for m in [sdk, trace, trulens_otel_semconv_trace]
        )

    @staticmethod
    def enable(
        session: experimental._WithExperimentalSettings,
    ):  # actually TruSession
        """Called when otel_tracing is enabled for session."""

        # Patch in Span ORM class into the session's database ORM.
        from trulens.core.database import sqlalchemy as sqlalchemy_db
        from trulens.experimental.otel_tracing.core.database import (
            orm as otel_orm,
        )
        from trulens.experimental.otel_tracing.core.database import (
            sqlalchemy as otel_sqlalchemy,
        )

        db = session.connector.db

        if not isinstance(db, sqlalchemy_db.SQLAlchemyDB):
            raise ValueError(
                "otel_tracing feature requires SQLAlchemyDB for database access."
            )

        print(f"Patching {db} with otel_tracing additions.")

        orm = db.orm
        tracing_orm = otel_orm.new_orm(orm.base)
        orm.Span = tracing_orm.Span

        # retrofit base SQLAlchemyDB with otel_tracing additions
        db.__class__ = otel_sqlalchemy._SQLAlchemyDB

        configure_mappers()

        # creates the new tables
        orm.metadata.create_all(db.engine)
