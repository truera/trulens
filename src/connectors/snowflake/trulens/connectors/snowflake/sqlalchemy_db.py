"""Snowflake-specific [SQLAlchemyDB][trulens.core.database.sqlalchemy.SQLAlchemyDB] subclass.

Encapsulates dialect-specific behavior that previously lived in
``trulens-core``:

* AUTOCOMMIT isolation_level workaround for ``snowflake-sqlalchemy >= 1.7.2``.
* NULL-result feedback insert workaround for the Snowflake stored-procedure
  connector.
* JSON path extraction via ``json_extract_path_text`` with quoted paths.
* Latency expression via ``timestampdiff``.
* Alembic ``DefaultImpl`` registration for the ``snowflake`` dialect.
"""

from __future__ import annotations

from typing import Any

from alembic.ddl.impl import DefaultImpl
from packaging.version import Version
import sqlalchemy as sa
from trulens.core.database.sqlalchemy import SQLAlchemyDB


class SnowflakeImpl(DefaultImpl):
    """Alembic DDL impl marker for the ``snowflake`` dialect."""

    __dialect__ = "snowflake"


class SnowflakeSQLAlchemyDB(SQLAlchemyDB):
    """[SQLAlchemyDB][trulens.core.database.sqlalchemy.SQLAlchemyDB] subclass
    that hosts all Snowflake-dialect-specific overrides."""

    def _apply_engine_param_overrides(self) -> None:
        snowflake_sqlalchemy_version = None
        try:
            import snowflake.sqlalchemy

            snowflake_sqlalchemy_version = snowflake.sqlalchemy.__version__
        except Exception:
            pass
        if (
            snowflake_sqlalchemy_version
            and Version(snowflake_sqlalchemy_version) >= Version("1.7.2")
            and "url" in self.engine_params
            and "snowflake" in self.engine_params["url"]
        ):
            temp_engine = sa.create_engine(**self.engine_params)
            try:
                if temp_engine.dialect.name == "snowflake":
                    self.engine_params.setdefault(
                        "isolation_level", "AUTOCOMMIT"
                    )
            finally:
                temp_engine.dispose()

    def _needs_null_feedback_hack(self, feedback_result: Any) -> bool:
        # The Snowflake stored-procedure connector cannot bind a None qmark to
        # a nullable numeric column on INSERT/UPDATE. Detect by dialect name
        # since the same DB class can be paired with non-Snowflake URLs in
        # tests.
        if self.engine is None:
            return False
        if self.engine.dialect.name != "snowflake":
            return False
        return getattr(feedback_result, "result", None) is None

    def _json_path_expr(self, column_obj: Any, path: str) -> Any:
        return sa.func.json_extract_path_text(column_obj, f'"{path}"')

    def _latency_expr(self) -> Any:
        return sa.func.avg(
            sa.func.timestampdiff(
                sa.text("SECOND"),
                self.orm.Event.start_timestamp,
                self.orm.Event.timestamp,
            )
        )
