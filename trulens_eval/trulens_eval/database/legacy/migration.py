"""
This is pre-sqlalchemy db migration. This file should not need changes. It is
here for backwards compatibility of oldest trulens-eval versions.
"""

import json
import logging
import shutil
import traceback
from typing import Callable, List
import uuid

import pydantic
from tqdm import tqdm

from trulens_eval.feedback import feedback as mod_feedback
from trulens_eval.schema import app as mod_app_schema
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import CLASS_INFO
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.pyschema import Method
from trulens_eval.utils.pyschema import Module
from trulens_eval.utils.pyschema import Obj

logger = logging.getLogger(__name__)
'''
How to make a db migrations:

1. Create a compatibility DB (checkout the last pypi rc branch https://github.com/truera/trulens/tree/releases/rc-trulens-eval-X.x.x/):
  In trulens/trulens_eval/tests/docs_notebooks/notebooks_to_test 
  remove any local dbs
    * rm rf default.sqlite
  run below notebooks (Making sure you also run with the same X.x.x version trulens-eval)
    * all_tools.ipynb # cp cp ../generated_files/all_tools.ipynb ./
    * llama_index_quickstart.ipynb # cp frameworks/llama_index/llama_index_quickstart.ipynb ./
    * langchain-retrieval-augmentation-with-trulens.ipynb # cp vector-dbs/pinecone/langchain-retrieval-augmentation-with-trulens.ipynb ./
    * Add any other notebooks you think may have possible breaking changes
  replace the last compatible db with this new db file
    * See the last COMPAT_VERSION: compatible version in leftmost below: migration_versions
    * mv default.sqlite trulens/trulens_eval/release_dbs/COMPAT_VERSION/default.sqlite

2. Do Migration coding
  * Update __init__.py with the new version
  * The upgrade methodology is determined by this datastructure
        upgrade_paths = {
            # from_version: (to_version,migrate_function)
            "0.1.2": ("0.2.0", migrate_0_1_2),
            "0.2.0": ("0.3.0", migrate_0_2_0)
        }
  * add your version to the version list:
      migration_versions: list = [YOUR VERSION HERE,...,"0.3.0", "0.2.0", "0.1.2"]


3. To Test
  * replace your db file with an old version db first and see if the tru.migrate_database() works.

4. Add a DB file for testing new breaking changes (Same as step 1: but with your new version) 
  * Do a sys.path.insert(0,TRULENS_PATH) to run with your version
'''


class VersionException(Exception):
    pass


MIGRATION_UNKNOWN_STR = "unknown[db_migration]"
migration_versions: List[str] = ["0.19.0"]


def _update_db_json_col(
    db, table: str, old_entry: tuple, json_db_col_idx: int, new_json: dict
):
    """Replaces an old json serialized db column with a migrated/new one.

    Args:
        db (DB): the db object

        table (str): the table to update (from the current DB)

        old_entry (tuple): the db tuple to update

        json_db_col_idx (int): the tuple idx to update
        
        new_json (dict): the new json object to be put in the D
    """
    migrate_record = list(old_entry)
    migrate_record[json_db_col_idx] = json.dumps(new_json)
    migrate_record = tuple(migrate_record)
    db._insert_or_replace_vals(table=table, vals=migrate_record)


def jsonlike_map(fval=None, fkey=None, fkeyval=None):
    if fval is None:
        fval = lambda x: x
    if fkey is None:
        fkey = lambda x: x
    if fkeyval is None:
        fkeyval = lambda x, y: (x, y)

    def walk(obj):
        if isinstance(obj, dict):
            ret = {}
            for k, v in obj.items():
                k = fkey(k)
                v = fval(walk(v))
                k, v = fkeyval(k, v)
                ret[k] = v
            return fval(ret)

        if isinstance(obj, (list, tuple)):
            return fval(type(obj)(fval(walk(v)) for v in obj))

        else:
            return fval(obj)

    return walk


def jsonlike_rename_key(old_key, new_key) -> Callable:

    def fkey(k):
        if k == old_key:
            logger.debug(f"key {old_key} -> {new_key}")
            return new_key
        else:
            return k

    return jsonlike_map(fkey=fkey)


def jsonlike_rename_value(old_val, new_val) -> Callable:

    def fval(v):
        if v == old_val:
            logger.debug(f"value {old_val} -> {new_val}")
            return new_val
        else:
            return v

    return jsonlike_map(fval=fval)


class UnknownClass(pydantic.BaseModel):

    def unknown_method(self):
        """
        This is a placeholder put into the database in place of methods whose
        information was not recorded in earlier versions of trulens.
        """


upgrade_paths = {
    # "from_version":("to_version", migrate_method)
    # "0.9.0": ("0.19.0", migrate_0_9_0)
}


def _parse_version(version_str: str) -> List[str]:
    """
    Takes a version string and returns a list of major, minor, patch.

    Args:
        - version_str (str): a version string

    Returns:
        list: [major, minor, patch] strings
    """
    return version_str.split(".")


def _get_compatibility_version(version: str) -> str:
    """
    Gets the db version that the pypi version is compatible with.

    Args:
        - version (str): a pypi version

    Returns:
        - str: a backwards compat db version
    """

    version_split = _parse_version(version)

    for m_version_str in migration_versions:
        for i, m_version_split in enumerate(_parse_version(m_version_str)):

            if int(version_split[i]) > int(m_version_split):
                return m_version_str

            elif int(version_split[i]) == int(m_version_split):
                if i == 2:  #patch version
                    return m_version_str
                # Can't make a choice here, move to next endian.
                continue

            else:
                # The m_version from m_version_str is larger than this version
                # check the next m_version.
                break


def _migration_checker(db, warn: bool = False) -> None:
    """
    Checks whether this db, if pre-populated, is comptible with this pypi
    version.

    Args:
        - db (DB): the db object to check
        - warn (bool, optional): if warn is False, then a migration issue will
          raise an exception, otherwise allow passing but only warn. Defaults to
          False.
    """
    meta = db.get_meta()

    _check_needs_migration(meta.trulens_version, warn=warn)


def commit_migrated_version(db, version: str) -> None:
    """After a successful migration, update the DB meta version

    Args:
        db (DB): the db object
        version (str): The version string to set this DB to
    """
    conn, c = db._connect()

    c.execute(
        f'''UPDATE {db.TABLE_META} 
                SET value = '{version}' 
                WHERE key='trulens_version'; 
            '''
    )
    conn.commit()


def _upgrade_possible(compat_version: str) -> bool:
    """
    Checks the upgrade paths to see if there is a valid migration from the DB to
    the current pypi version

    Args:
        - compat_version (str): the current db version.

    Returns:
        - bool: True if there is an upgrade path. False if not.
    """
    while compat_version in upgrade_paths:
        compat_version = upgrade_paths[compat_version][0]
    return compat_version == migration_versions[0]


def _check_needs_migration(version: str, warn=False) -> None:
    """
    Checks whether the from DB version can be updated to the current DB version.

    Args:
        - version (str): the pypi version
        - warn (bool, optional): if warn is False, then a migration issue will
          raise an exception, otherwise allow passing but only warn. Defaults to
          False.
    """
    compat_version = _get_compatibility_version(version)

    if migration_versions.index(compat_version) > 0:

        if _upgrade_possible(compat_version):
            msg = (
                f"Detected that your db version {version} is from an older release that is incompatible with this release. "
                f"You can either reset your db with `tru.reset_database()`, "
                f"or you can initiate a db migration with `tru.migrate_database()`"
            )
        else:
            msg = (
                f"Detected that your db version {version} is from an older release that is incompatible with this release and cannot be migrated. "
                f"Reset your db with `tru.reset_database()`"
            )
        if warn:
            print(f"Warning! {msg}")
        else:
            raise VersionException(msg)


saved_db_locations = {}


def _serialization_asserts(db) -> None:
    """
    After a successful migration, Do some checks if serialized jsons are loading
    properly.

    Args:
        db (DB): the db object
    """
    global saved_db_locations
    conn, c = db._connect()
    SAVED_DB_FILE_LOC = saved_db_locations[db.filename]
    validation_fail_advice = (
        f"Please open a ticket on trulens github page including details on the old and new trulens versions. "
        f"The migration completed so you can still proceed; but stability is not guaranteed. "
        f"Your original DB file is saved here: {SAVED_DB_FILE_LOC} and can be used with the previous version, or you can `tru.reset_database()`"
    )

    for table in db.TABLES:
        c.execute(f"""PRAGMA table_info({table});
                """)
        columns = c.fetchall()
        for col_idx, col in tqdm(
                enumerate(columns),
                desc=f"Validating clean migration of table {table}"):
            col_name_idx = 1
            col_name = col[col_name_idx]
            # This is naive for now...
            if "json" in col_name:
                c.execute(f"""SELECT * FROM {table}""")
                rows = c.fetchall()
                for row in rows:
                    try:
                        if row[col_idx] == MIGRATION_UNKNOWN_STR:
                            continue

                        test_json = json.loads(row[col_idx])
                        # special implementation checks for serialized classes
                        if 'implementation' in test_json:
                            try:
                                FunctionOrMethod.model_validate(
                                    test_json['implementation']
                                ).load()
                            except ImportError:
                                # Import error is not a migration problem.
                                # It signals that the function cannot be used for deferred evaluation.
                                pass

                        if col_name == "record_json":
                            mod_record_schema.Record.model_validate(test_json)
                        elif col_name == "cost_json":
                            mod_base_schema.Cost.model_validate(test_json)
                        elif col_name == "perf_json":
                            mod_base_schema.Perf.model_validate(test_json)
                        elif col_name == "calls_json":
                            for record_app_call_json in test_json['calls']:
                                mod_feedback_schema.FeedbackCall.model_validate(
                                    record_app_call_json
                                )
                        elif col_name == "feedback_json":
                            mod_feedback_schema.FeedbackDefinition.model_validate(
                                test_json
                            )
                        elif col_name == "app_json":
                            mod_app_schema.AppDefinition.model_validate(
                                test_json
                            )
                        else:
                            # If this happens, trulens needs to add a migration

                            raise VersionException(
                                f"serialized column migration not implemented: {col_name}. {validation_fail_advice}"
                            )
                    except Exception as e:
                        tb = traceback.format_exc()

                        raise VersionException(
                            f"Migration failed on {table} {col_name} {row[col_idx]}.\n\n{tb}\n\n{validation_fail_advice}"
                        ) from e


def migrate(db) -> None:
    """Migrate a db to the compatible version of this pypi version

    Args:
        db (DB): the db object
    """
    # NOTE TO DEVELOPER: If this method fails: It's likely you made a db
    # breaking change. Follow these steps to add a compatibility change
    # - Update the __init__ version to the next one (if not already)
    # - In this file: add that version to `migration_versions` variable`
    # - Add the migration step in `upgrade_paths` of the form
    #   `from_version`:(`to_version_you_just_created`, `migration_function`)
    # - AFTER YOU PASS TESTS - add your newest db into
    #   `release_dbs/<version_you_just_created>/default.sqlite`
    #   - This is created by running the all_tools and llama_quickstart from a
    #     fresh db (you can `rm -rf` the sqlite file )
    #   - TODO: automate this step
    original_db_file = db.filename
    global saved_db_locations

    saved_db_file = original_db_file.parent / f"{original_db_file.name}_saved_{uuid.uuid1()}"
    saved_db_locations[original_db_file] = saved_db_file
    shutil.copy(original_db_file, saved_db_file)
    print(
        f"Saved original db file: `{original_db_file}` to new file: `{saved_db_file}`"
    )

    version = db.get_meta().trulens_version
    from_compat_version = _get_compatibility_version(version)
    while from_compat_version in upgrade_paths:
        to_compat_version, migrate_fn = upgrade_paths[from_compat_version]
        migrate_fn(db=db)
        commit_migrated_version(db=db, version=to_compat_version)
        from_compat_version = to_compat_version

    print("DB Migration complete!")
    _serialization_asserts(db)
    print("DB Validation complete!")
