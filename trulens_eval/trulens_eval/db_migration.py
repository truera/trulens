import json
import shutil
import traceback
import uuid

from tqdm import tqdm

from trulens_eval.schema import AppDefinition
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackCall
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import Perf
from trulens_eval.schema import Record
from trulens_eval.util import FunctionOrMethod


class VersionException(Exception):
    pass


MIGRATION_UNKNOWN_STR = "unknown[db_migration]"
migration_versions: list = ["0.3.0", "0.2.0", "0.1.2"]


def _update_db_json_col(
    db, table: str, old_entry: tuple, json_db_col_idx: int, new_json: dict
):
    """Replaces an old json serialized db column with a migrated/new one

    Args:
        db (DB): the db object
        table (str): the table to update (from the current DB)
        old_entry (tuple): the db tuple to update
        json_db_col_idx (int): the tuple idx to update
        new_json (dict): the new json object to be put in the DB
    """
    migrate_record = list(old_entry)
    migrate_record[json_db_col_idx] = json.dumps(new_json)
    migrate_record = tuple(migrate_record)
    db._insert_or_replace_vals(table=table, vals=migrate_record)


def migrate_0_2_0(db):
    """
    Migrates from 0.2.0 to 0.3.0
    Args:
        db (DB): the db object
    """

    conn, c = db._connect()
    c.execute(
        f"""SELECT * FROM records"""
    )  # Use hardcode names as versions could go through name change
    rows = c.fetchall()
    json_db_col_idx = 7

    def _replace_cost_none_vals(new_json):
        if new_json['n_tokens'] is None:
            new_json['n_tokens'] = 0

        if new_json['cost'] is None:
            new_json['cost'] = 0.0
        return new_json

    for old_entry in tqdm(rows, desc="Migrating Records DB 0.2.0 to 0.3.0"):
        new_json = _replace_cost_none_vals(
            json.loads(old_entry[json_db_col_idx])
        )
        _update_db_json_col(
            db=db,
            table=
            "records",  # Use hardcode names as versions could go through name change
            old_entry=old_entry,
            json_db_col_idx=json_db_col_idx,
            new_json=new_json
        )

    c.execute(f"""SELECT * FROM feedbacks""")
    rows = c.fetchall()
    json_db_col_idx = 9
    for old_entry in tqdm(rows, desc="Migrating Feedbacks DB 0.2.0 to 0.3.0"):
        new_json = _replace_cost_none_vals(
            json.loads(old_entry[json_db_col_idx])
        )
        _update_db_json_col(
            db=db,
            table="feedbacks",
            old_entry=old_entry,
            json_db_col_idx=json_db_col_idx,
            new_json=new_json
        )

    c.execute(f"""SELECT * FROM feedback_defs""")
    rows = c.fetchall()
    json_db_col_idx = 1
    for old_entry in tqdm(rows,
                          desc="Migrating FeedbackDefs DB 0.2.0 to 0.3.0"):
        new_json = json.loads(old_entry[json_db_col_idx])
        if 'implementation' in new_json:
            new_json['implementation']['obj']['cls']['module'][
                'module_name'] = new_json['implementation']['obj']['cls'][
                    'module']['module_name'].replace(
                        "tru_feedback", "feedback"
                    )
            if 'init_kwargs' in new_json['implementation']['obj']:
                new_json['implementation']['obj']['init_bindings'] = {
                    'args': (),
                    'kwargs': new_json['implementation']['obj']['init_kwargs']
                }
                del new_json['implementation']['obj']['init_kwargs']
        _update_db_json_col(
            db=db,
            table="feedback_defs",
            old_entry=old_entry,
            json_db_col_idx=json_db_col_idx,
            new_json=new_json
        )
    conn.commit()


def migrate_0_1_2(db):
    """
    Migrates from 0.1.2 to 0.2.0
    Args:
        db (DB): the db object
    """
    conn, c = db._connect()

    c.execute(
        f"""ALTER TABLE records
        RENAME COLUMN chain_id TO app_id;
        """
    )
    c.execute(
        f"""ALTER TABLE records
        ADD perf_json TEXT NOT NULL 
        DEFAULT "{MIGRATION_UNKNOWN_STR}";"""
    )

    c.execute(f"""ALTER TABLE feedbacks
        DROP COLUMN chain_id;""")

    c.execute(
        f"""SELECT * FROM records"""
    )  # Use hardcode names as versions could go through name change
    rows = c.fetchall()
    json_db_col_idx = 4
    for old_entry in tqdm(rows, desc="Migrating Records DB 0.1.2 to 0.2.0"):
        new_json = json.loads(old_entry[json_db_col_idx])
        new_json['app_id'] = new_json['chain_id']
        del new_json['chain_id']
        for calls_json in new_json['calls']:
            calls_json['stack'] = calls_json['chain_stack']
            del calls_json['chain_stack']

        _update_db_json_col(
            db=db,
            table=
            "records",  # Use hardcode names as versions could go through name change
            old_entry=old_entry,
            json_db_col_idx=json_db_col_idx,
            new_json=new_json
        )

    c.execute(f"""SELECT * FROM chains""")
    rows = c.fetchall()
    json_db_col_idx = 1
    for old_entry in tqdm(rows, desc="Migrating Apps DB 0.1.2 to 0.2.0"):
        new_json = json.loads(old_entry[json_db_col_idx])
        new_json['app_id'] = new_json['chain_id']
        del new_json['chain_id']
        new_json['root_class'] = {
            'name': 'Unknown_class',
            'module':
                {
                    'package_name': MIGRATION_UNKNOWN_STR,
                    'module_name': MIGRATION_UNKNOWN_STR
                },
            'bases': None
        }
        new_json['feedback_mode'] = new_json['feedback_mode'].replace(
            'chain', 'app'
        )
        del new_json['db']
        _update_db_json_col(
            db=db,
            table="apps",
            old_entry=old_entry,
            json_db_col_idx=json_db_col_idx,
            new_json=new_json
        )

    conn.commit()


upgrade_paths = {
    "0.1.2": ("0.2.0", migrate_0_1_2),
    "0.2.0": ("0.3.0", migrate_0_2_0)
}


def _parse_version(version_str: str) -> list:
    """takes a version string and returns a list of major, minor, patch

    Args:
        version_str (str): a version string

    Returns:
        list: [major, minor, patch]
    """
    return version_str.split(".")


def _get_compatibility_version(version: str) -> str:
    """Gets the db version that the pypi version is compatible with

    Args:
        version (str): a pypi version

    Returns:
        str: a backwards compat db version
    """
    version_split = _parse_version(version)
    for m_version_str in migration_versions:
        for i, m_version_split in enumerate(_parse_version(m_version_str)):
            if version_split[i] > m_version_split:
                return m_version_str
            elif version_split[i] == m_version_split:
                if i == 2:  #patch version
                    return m_version_str
                # Can't make a choice here, move to next endian
                continue
            else:
                # the m_version from m_version_str is larger than this version. check the next m_version
                break


def _migration_checker(db, warn=False) -> None:
    """Checks whether this db, if pre-populated, is comptible with this pypi version

    Args:
        db (DB): the db object to check
        warn (bool, optional): if warn is False, then a migration issue will raise an exception, otherwise allow passing but only warn. Defaults to False.
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
    """Checks the upgrade paths to see if there is a valid migration from the DB to the current pypi version

    Args:
        compat_version (str): the current db version

    Returns:
        bool: True if there is an upgrade path. False if not.
    """
    while compat_version in upgrade_paths:
        compat_version = upgrade_paths[compat_version][0]
    return compat_version == migration_versions[0]


def _check_needs_migration(version: str, warn=False) -> None:
    """Checks whether the from DB version can be updated to the current DB version.

    Args:
        version (str): the pypi version
        warn (bool, optional): if warn is False, then a migration issue will raise an exception, otherwise allow passing but only warn. Defaults to False.
    """
    compat_version = _get_compatibility_version(version)
    if migration_versions.index(compat_version) > 0:
        if _upgrade_possible(compat_version):
            msg = f"Detected that your db version {version} is from an older release that is incompatible with this release. you can either reset your db with `tru.reset_database()`, or you can initiate a db migration with `tru.migrate_database()`"
        else:
            msg = f"Detected that your db version {version} is from an older release that is incompatible with this release and cannot be migrated. Reset your db with `tru.reset_database()`"
        if warn:
            print(f"Warning! {msg}")
        else:
            raise VersionException(msg)


saved_db_locations = {}


def _serialization_asserts(db) -> None:
    """After a successful migration, Do some checks if serialized jsons are loading properly

    Args:
        db (DB): the db object
    """
    global saved_db_locations
    conn, c = db._connect()
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
                            FunctionOrMethod.pick(
                                **(test_json['implementation'])
                            ).load()

                        if col_name == "record_json":
                            Record(**test_json)
                        elif col_name == "cost_json":
                            Cost(**test_json)
                        elif col_name == "perf_json":
                            Perf(**test_json)
                        elif col_name == "calls_json":
                            for record_app_call_json in test_json['calls']:
                                FeedbackCall(**record_app_call_json)
                        elif col_name == "feedback_json":
                            FeedbackDefinition(**test_json)
                        elif col_name == "app_json":
                            AppDefinition(**test_json)
                        else:
                            # If this happens, trulens needs to add a migration
                            SAVED_DB_FILE_LOC = saved_db_locations[db.filename]
                            raise VersionException(
                                f"serialized column migration not implemented. Please open a ticket on trulens github page including details on the old and new trulens versions. Your original DB file is saved here: {SAVED_DB_FILE_LOC}"
                            )
                    except Exception as e:
                        tb = traceback.format_exc()
                        raise VersionException(
                            f"Migration failed on {table} {col_name} {row[col_idx]}.\n\n{tb}"
                        )


def migrate(db) -> None:
    """Migrate a db to the compatible version of this pypi version

    Args:
        db (DB): the db object
    """
    # NOTE TO DEVELOPER: If this method fails: It's likely you made a db breaking change.
    # Follow these steps to add a compatibility change
    # - Update the __init__ version to the next one (if not already)
    # - In this file: add that version to `migration_versions` variable`
    # - Add the migration step in `upgrade_paths` of the form `from_version`:(`to_version_you_just_created`, `migration_function`)
    # - AFTER YOU PASS TESTS - add your newest db into `release_dbs/<version_you_just_created>/default.sqlite`
    #   - This is created by running the all_tools and llama_quickstart from a fresh db (you can `rm -rf` the sqlite file )
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

    _serialization_asserts(db)
    print("DB Migration complete!")
