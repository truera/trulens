from __future__ import annotations

import json
import traceback
from typing import Callable, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session
from trulens.core.database import migrations as db_migrations
from trulens.core.database.base import DB
from trulens.core.database.legacy import migration as legacy_migration
from trulens.core.schema import app as app_schema
from trulens.core.schema import base as base_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.utils import pyschema as pyschema_utils

sql_alchemy_migration_versions: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
"""DB versions."""

sqlalchemy_upgrade_paths: Dict[int, Tuple[int, Callable[[DB]]]] = {
    # Dict Structure:
    # from_version: (to_version, migrate_method)
    # Example:
    # 1: (2, migrate_alembic_1_to_2)
}
"""A DAG of upgrade functions to get to most recent DB."""


def _get_sql_alchemy_compatibility_version(version: int) -> int:
    """Gets the last compatible version of a DB that needed data migration.

    Args:
        version: The alembic version

    Returns:
        int: An alembic version of the oldest compatible DB
    """

    for candidate_version in sorted(
        sql_alchemy_migration_versions, reverse=True
    ):
        if candidate_version <= version:
            return candidate_version
    raise ValueError(
        f"The version given {version} is not compatible with any known version! Known versions: {sql_alchemy_migration_versions}"
    )


def _sql_alchemy_serialization_asserts(db: DB) -> None:
    """Checks that data migrated JSONs can be deserialized from DB to Python objects.

    Args:
        db (DB): The database object

    Raises:
        VersionException: raises if a serialization fails
    """
    session = Session(db.engine)

    import inspect

    # from trulens.core.database import orm
    # Dynamically check the orm classes since these could change version to version
    for _, orm_obj in inspect.getmembers(db.orm):
        # Check only classes
        if inspect.isclass(orm_obj):
            mod_check = str(orm_obj).split(".")

            # Check only orm defined classes
            if (
                len(mod_check) > 2 and "orm" == mod_check[-2]
            ):  # <class mod.mod.mod.orm.SQLORM>
                stmt = select(orm_obj)

                # for each record in this orm table
                for db_record in session.scalars(stmt).all():
                    # for each column in the record
                    for attr_name in db_record.__dict__:
                        # Check only json columns
                        if "_json" in attr_name:
                            db_json_str = getattr(db_record, attr_name)
                            if (
                                db_json_str
                                == legacy_migration.MIGRATION_UNKNOWN_STR
                            ):
                                continue

                            # Do not check Nullables
                            if db_json_str is not None:
                                # Test deserialization
                                test_json = json.loads(
                                    getattr(db_record, attr_name)
                                )

                                # special implementation checks for serialized classes
                                if "implementation" in test_json:
                                    try:
                                        pyschema_utils.FunctionOrMethod.model_validate(
                                            test_json["implementation"]
                                        ).load()
                                    except ImportError:
                                        # Import error is not a migration problem.
                                        # It signals that the function cannot be used for deferred evaluation.
                                        pass

                                if attr_name == "record_json":
                                    record_schema.Record.model_validate(
                                        test_json
                                    )
                                elif attr_name == "cost_json":
                                    base_schema.Cost.model_validate(test_json)
                                elif attr_name == "perf_json":
                                    base_schema.Perf.model_validate(test_json)
                                elif attr_name == "calls_json":
                                    for record_app_call_json in test_json[
                                        "calls"
                                    ]:
                                        feedback_schema.FeedbackCall.model_validate(
                                            record_app_call_json
                                        )
                                elif attr_name == "feedback_json":
                                    feedback_schema.FeedbackDefinition.model_validate(
                                        test_json
                                    )
                                elif attr_name == "app_json":
                                    app_schema.AppDefinition.model_validate(
                                        test_json
                                    )
                                else:
                                    # If this happens, trulens needs to add a migration
                                    raise legacy_migration.VersionException(
                                        f"serialized column migration not implemented: {attr_name}."
                                    )


def data_migrate(db: DB, from_version: Optional[str]):
    """Makes any data changes needed for upgrading from the from_version to the
    current version.

    Args:
        db: The database instance.

        from_version: The version to migrate data from.

    Raises:
        VersionException: Can raise a migration or validation upgrade error.
    """

    if from_version is None:
        sql_alchemy_from_version = 1
    else:
        sql_alchemy_from_version = int(from_version)
    from_compat_version = _get_sql_alchemy_compatibility_version(
        sql_alchemy_from_version
    )
    to_compat_version = None
    fail_advice = "Please open a ticket on trulens github page including this error message. The migration completed so you can still proceed; but stability is not guaranteed. If needed, you can `TruSession.reset_database()`"

    try:
        while from_compat_version in sqlalchemy_upgrade_paths:
            to_compat_version, migrate_fn = sqlalchemy_upgrade_paths[
                from_compat_version
            ]
            migrate_fn(db)
            from_compat_version = to_compat_version
        print("DB Migration complete!")
    except Exception as e:
        tb = traceback.format_exc()
        current_revision = db_migrations.DbRevisions.load(db.engine).current
        raise legacy_migration.VersionException(
            f"Migration failed on {db} from db version - {from_version} on step: {str(to_compat_version)}. The attempted DB version is {current_revision} \n\n{tb}\n\n{fail_advice}"
        ) from e
    try:
        _sql_alchemy_serialization_asserts(db)
        print("DB Validation complete!")
    except Exception as e:
        tb = traceback.format_exc()
        current_revision = db_migrations.DbRevisions.load(db.engine).current
        raise legacy_migration.VersionException(
            f"Validation failed on {db} from db version - {from_version} on step: {str(to_compat_version)}. The attempted DB version is {current_revision} \n\n{tb}\n\n{fail_advice}"
        ) from e
