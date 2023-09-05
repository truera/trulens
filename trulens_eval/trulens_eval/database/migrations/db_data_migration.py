import json
import traceback
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from trulens_eval.database.migrations import DbRevisions
from trulens_eval.db_migration import MIGRATION_UNKNOWN_STR
from trulens_eval.db_migration import VersionException
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackCall
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import Perf
from trulens_eval.schema import Record
from trulens_eval.utils.pyschema import FunctionOrMethod

# Keeps track of any db versions that need data migration
# Should be most recent is the leftmost in the list
sql_alchemy_migration_versions: List[str] = ["1"]

# A DAG of upgrade functions to get to most recent DB.
sqlalchemy_upgrade_paths = {
    # Dict Structure:
    # "from_version":("to_version", migrate_method)

    # Example:
    # "1":("2"), migrate_alembic_1_to_2
}


def _get_sql_alchemy_compatibility_version(version: str) -> str:
    """Gets the last compatible version of a DB that needed data migration

    Args:
        version (str): The alembic version

    Returns:
        str: An alembic version of the oldest compatible DB
    """
    compat_version = int(sql_alchemy_migration_versions[-1])
    for candidate_version in sql_alchemy_migration_versions:
        candidate_version_int = int(candidate_version)
        if candidate_version_int <= int(
                version) and candidate_version_int > compat_version:
            compat_version = candidate_version_int
    return compat_version


def _sql_alchemy_serialization_asserts(db: "DB") -> None:
    """Checks that data migrated JSONs can be deserialized from DB to Python objects.

    Args:
        db (DB): The database object

    Raises:
        VersionException: raises if a serialization fails
    """
    session = Session(db.engine)

    import inspect

    from trulens_eval.database import orm

    # Dynamically check the orm classes since these could change version to version
    for _, orm_obj in inspect.getmembers(orm):

        # Check only classes
        if inspect.isclass(orm_obj):
            mod_check = str(orm_obj).split(".")

            # Check only orm defined classes
            if len(mod_check) > 2 and "orm" == mod_check[
                    -2]:  # <class mod.mod.mod.orm.SQLORM>
                stmt = select(orm_obj)

                # for each record in this orm table
                for db_record in session.scalars(stmt).all():

                    # for each column in the record
                    for attr_name in db_record.__dict__:

                        # Check only json columns
                        if "_json" in attr_name:
                            db_json_str = getattr(db_record, attr_name)
                            if db_json_str == MIGRATION_UNKNOWN_STR:
                                continue

                            # Do not check Nullables
                            if db_json_str is not None:

                                # Test deserialization
                                test_json = json.loads(
                                    getattr(db_record, attr_name)
                                )

                                # special implementation checks for serialized classes
                                if 'implementation' in test_json:
                                    try:
                                        FunctionOrMethod.pick(
                                            **(test_json['implementation'])
                                        ).load()
                                    except ImportError:
                                        # Import error is not a migration problem.
                                        # It signals that the function cannot be used for deferred evaluation.
                                        pass

                                if attr_name == "record_json":
                                    Record(**test_json)
                                elif attr_name == "cost_json":
                                    Cost(**test_json)
                                elif attr_name == "perf_json":
                                    Perf(**test_json)
                                elif attr_name == "calls_json":
                                    for record_app_call_json in test_json[
                                            'calls']:
                                        FeedbackCall(**record_app_call_json)
                                elif attr_name == "feedback_json":
                                    FeedbackDefinition(**test_json)
                                elif attr_name == "app_json":
                                    AppDefinition(**test_json)
                                else:
                                    # If this happens, trulens needs to add a migration
                                    raise VersionException(
                                        f"serialized column migration not implemented: {attr_name}."
                                    )


def data_migrate(db: "DB", from_version: str):
    """Makes any data changes needed for upgrading from the from_version

    Args:
        db (DB): The Database Object
        from_version (str): The current version

    Raises:
        VersionException: Can raise a migration or validation upgrade error
    """

    if from_version is None:
        sql_alchemy_from_version = "1"
    else:
        sql_alchemy_from_version = from_version
    from_compat_version = _get_sql_alchemy_compatibility_version(
        sql_alchemy_from_version
    )
    to_compat_version = None
    fail_advice = f"Please open a ticket on trulens github page including this error message. The migration completed so you can still proceed; but stability is not guaranteed. If needed, you can `tru.reset_database()`"

    try:
        while from_compat_version in sqlalchemy_upgrade_paths:
            to_compat_version, migrate_fn = sqlalchemy_upgrade_paths[
                from_compat_version]

            migrate_fn(db=db)
            from_compat_version = to_compat_version

        print("DB Migration complete!")
    except Exception:
        tb = traceback.format_exc()
        current_revision = DbRevisions.load(db.engine).current
        raise VersionException(
            f"Migration failed on {db} from db version - {from_version} on step: {str(to_compat_version)}. The attempted DB version is {current_revision} \n\n{tb}\n\n{fail_advice}"
        )
    try:
        _sql_alchemy_serialization_asserts(db)
        print("DB Validation complete!")
    except Exception:
        tb = traceback.format_exc()
        current_revision = DbRevisions.load(db.engine).current
        raise VersionException(
            f"Validation failed on {db} from db version - {from_version} on step: {str(to_compat_version)}. The attempted DB version is {current_revision} \n\n{tb}\n\n{fail_advice}"
        )
