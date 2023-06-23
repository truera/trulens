#TODO: if backwards compat fails -> tell them to modify this file
from tqdm import tqdm
import json
import traceback

from trulens_eval.schema import Record, Cost, Perf, FeedbackDefinition, AppDefinition, FeedbackCall
from trulens_eval.util import FunctionOrMethod

class VersionException(Exception):
    pass

MIGRATION_UNKNOWN_STR="unknown[db_migration]"
migration_versions: list = ["0.3.0", "0.2.0", "0.1.2"]
def _update_db_json_col(db, table, old_entry, json_db_col_idx, new_json):
    migrate_record = list(old_entry)
    migrate_record[json_db_col_idx] = json.dumps(new_json) 
    migrate_record = tuple(migrate_record)      
    db._insert_or_replace_vals(table=table, vals=migrate_record)  

def migrate_0_2_0(db):
    conn, c = db._connect()
    c.execute(f"""SELECT * FROM feedback_defs""")
    rows = c.fetchall()
    json_db_col_idx = 1
    for old_entry in tqdm(rows,desc="Migrating FeedbackDefs DB"):
        new_json = json.loads(old_entry[json_db_col_idx])
        if 'implementation' in new_json:
            new_json['implementation']['obj']['cls']['module']['module_name'] = new_json['implementation']['obj']['cls']['module']['module_name'].replace("tru_feedback", "feedback")
        _update_db_json_col(db=db, table=db.TABLE_FEEDBACK_DEFS, old_entry=old_entry, json_db_col_idx=json_db_col_idx, new_json=new_json)
    conn.commit()
def migrate_0_1_2(db):
    conn, c = db._connect()

    c.execute(f"""ALTER TABLE records
        RENAME COLUMN chain_id TO app_id;
        """)
    c.execute(f"""ALTER TABLE records
        ADD perf_json TEXT NOT NULL 
        DEFAULT "{MIGRATION_UNKNOWN_STR}";""")
    
    c.execute(f"""ALTER TABLE feedbacks
        DROP COLUMN chain_id;""")

    c.execute(f"""SELECT * FROM records""")
    rows = c.fetchall()
    json_db_col_idx = 4
    for old_entry in tqdm(rows,desc="Migrating Records DB"):
        new_json = json.loads(old_entry[json_db_col_idx])
        new_json['app_id'] = new_json['chain_id']
        del new_json['chain_id']
        for calls_json in new_json['calls']:
            calls_json['stack']=calls_json['chain_stack']
            del calls_json['chain_stack']
        
        _update_db_json_col(db=db, table=db.TABLE_RECORDS, old_entry=old_entry, json_db_col_idx=json_db_col_idx, new_json=new_json)
        
        
    c.execute(f"""SELECT * FROM chains""")
    rows = c.fetchall()
    json_db_col_idx = 1
    for old_entry in tqdm(rows,desc="Migrating Apps DB"):
        new_json = json.loads(old_entry[json_db_col_idx])
        new_json['app_id'] = new_json['chain_id']
        del new_json['chain_id']
        new_json['root_class'] = {'name': 'Unknown_class', 'module': {'package_name': MIGRATION_UNKNOWN_STR, 'module_name': MIGRATION_UNKNOWN_STR}, 'bases': None}
        new_json['feedback_mode'] = new_json['feedback_mode'].replace('chain', 'app')
        del new_json['db']
        _update_db_json_col(db=db, table=db.TABLE_APPS, old_entry=old_entry, json_db_col_idx=json_db_col_idx, new_json=new_json)    

    conn.commit()
    
upgrade_paths = {
            "0.1.2":("0.2.0", migrate_0_1_2),
            "0.2.0":("0.3.0", migrate_0_2_0)
        }
    

def _parse_version(version_str):
    return version_str.split(".")

def _get_compatibility_version(version):
    version_split = _parse_version(version)
    for m_version_str in migration_versions:
        for i, m_version_split in enumerate(_parse_version(m_version_str)):
            if version_split[i] > m_version_split:
                return m_version_str
            elif version_split[i] == m_version_split:
                if i==2: #patch version
                    return m_version_str
                # Can't make a choice here, move to next endian
                continue
            else:
                # the m_version from m_version_str is larger than this version. check the next m_version
                break

def _migration_checker(db, warn=False):
    meta = db.get_meta()
    check_needs_migration(meta.trulens_version, warn=warn)

def commit_migrated_version(db, version) -> None:
    conn, c = db._connect()
    
    c.execute(
            f'''UPDATE {db.TABLE_META} 
                SET value = '{version}' 
                WHERE key='trulens_version'; 
            '''
        )
    conn.commit()

def _upgrade_possible(compat_version):
    while compat_version in upgrade_paths:
        compat_version = upgrade_paths[compat_version][0]
    return compat_version == migration_versions[0]

def check_needs_migration(version, warn=False):
    compat_version = _get_compatibility_version(version)
    if migration_versions.index(compat_version) > 0:
        if _upgrade_possible(compat_version):
            msg=f"Detected that your db version {version} is from an older release that is incompatible with this release. you can either reset your db with `tru.reset_database()`, or you can initiate a db migration with `tru.migrate_database()`"
        else:
            msg = f"Detected that your db version {version} is from an older release that is incompatible with this release and cannot be migrated. Reset your db with `tru.reset_database()`"
        if warn:
            print(f"Warning! {msg}")
        else:
            raise VersionException(msg)


def _serialization_asserts(db):
    conn, c = db._connect()
    for table in db.TABLES:
        c.execute(f"""PRAGMA table_info({table});
                """)
        columns = c.fetchall()
        for col_idx, col in tqdm(enumerate(columns), desc=f"Validating clean migration of table {table}"):
            col_name_idx = 1
            col_name=col[col_name_idx]
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
                            TODO_FILE_LOC = "TODO_FILE_LOC"
                            raise VersionException(f"serialized column migration not implemented. Please open a ticket on trulens github page including details on the old and new trulens versions. Your original DB file is saved here: {TODO_FILE_LOC}")
                    except Exception as e:
                        tb = traceback.format_exc()
                        raise VersionException(f"Migration failed on {table} {col_name} {row[col_idx]}.\n\n{tb}")

        

def migrate(db):
    # TODO: Save original DB
    version = db.get_meta().trulens_version
    from_compat_version = _get_compatibility_version(version)
    while from_compat_version in upgrade_paths:
        to_compat_version, migrate_fn = upgrade_paths[from_compat_version]
        migrate_fn(db=db)
        commit_migrated_version(db=db, version=to_compat_version)
        from_compat_version=to_compat_version
    
    _serialization_asserts(db)
    print("DB Migration complete!")

