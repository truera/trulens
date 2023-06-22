#TODO: if backwards compat fails -> tell them to modify this file
from tqdm import tqdm
import json

from trulens_eval.schema import Record, Cost, Perf, RecordAppCall, FeedbackDefinition, AppDefinition

class VersionException(Exception):
    pass

MIGRATION_UNKNOWN_STR="unknown[db_migration]"
migration_versions: list = ["0.3.0", "0.2.0", "0.1.2"]
def migrate_0_2_0(db):
    pass
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
    
    #TODO: REMOVE THESE DEBUGS
    c.execute(f"""SELECT * FROM feedbacks""")
    rows = c.fetchall()
    print(f"FEEDBACK {rows}")
    c.execute(f"""SELECT * FROM feedback_defs""")
    rows = c.fetchall()
    print(f"FEEDBACKDEFS {rows}")

    
    c.execute(f"""SELECT * FROM records""")
    rows = c.fetchall()
    for old_record in tqdm(rows,desc="Migrating Records DB"):

        record_json_db_idx = 4
        record_json = json.loads(old_record[record_json_db_idx])
        record_json['app_id'] = record_json['chain_id']
        del record_json['chain_id']
        for calls_json in record_json['calls']:
            calls_json['stack']=calls_json['chain_stack']
            del calls_json['chain_stack']
        
        migrate_record = list(old_record)
        migrate_record[record_json_db_idx] = json.dumps(record_json) 
        migrate_record = tuple(migrate_record)
        
        db._insert_or_replace_vals(table=db.TABLE_RECORDS, vals=migrate_record)
        
    c.execute(f"""SELECT * FROM chains""")
    rows = c.fetchall()
    for old_chain in tqdm(rows,desc="Migrating Apps DB"):
        app_json_db_idx = 1
        app_json = json.loads(old_chain[app_json_db_idx])
        app_json['app_id'] = app_json['chain_id']
        del app_json['chain_id']
        app_json['root_class'] = {'name': 'Unknown_class', 'module': {'package_name': MIGRATION_UNKNOWN_STR, 'module_name': MIGRATION_UNKNOWN_STR}, 'bases': None}
        
        vals = (old_chain[0], json.dumps(app_json))
        db._insert_or_replace_vals(table=db.TABLE_APPS, vals=vals)
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
                    if row[col_idx] == MIGRATION_UNKNOWN_STR:
                        continue
                    
                    test_json = json.loads(row[col_idx])
                    
                    if col_name == "record_json":
                        Record(**test_json)
                    elif col_name == "cost_json":
                        Cost(**test_json)
                    elif col_name == "perf_json":
                        Perf(**test_json)
                    elif col_name == "calls_json":
                        for record_app_call_json in test_json['calls']:
                            print(record_app_call_json)
                            RecordAppCall(**record_app_call_json)
                    elif col_name == "feedback_json":
                        FeedbackDefinition(**test_json)
                    elif col_name == "app_json":
                        AppDefinition(**test_json)
                    else:
                        # If this happens, trulens needs to add a migration
                        TODO_FILE_LOC = "TODO_FILE_LOC"
                        raise VersionException(f"Migration failed on {table} {col_name} {row[col_idx]}. Please open a ticket on trulens github page including details on the old and new trulens versions. Your original DB file is saved here: {TODO_FILE_LOC}")
    
        

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

