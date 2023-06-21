#TODO: if backwards compat fails -> tell them to modify this file
from tqdm import tqdm
import json

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
    
    c.execute(f"""SELECT * FROM chains""")
    rows = c.fetchall()
    for old_chain in tqdm(rows,desc="Migrating Apps DB"):
        app_json = json.loads(old_chain[1])
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


def migrate(db):
    version = db.get_meta().trulens_version
    from_compat_version = _get_compatibility_version(version)
    while from_compat_version in upgrade_paths:
        to_compat_version, migrate_fn = upgrade_paths[from_compat_version]
        migrate_fn(db=db)
        commit_migrated_version(db=db, version=to_compat_version)
        from_compat_version=to_compat_version
    print("DB Migration complete!")

