# 🕸✨ Database Migration

When upgrading _TruLens-Eval_, it may sometimes be required to migrade the
database to incorporate changes in existing database created from the previously
installed version. The changes to database schemas is handled by
[Alembic](https://github.com/sqlalchemy/alembic/) while some data changes are
handled by converters in [the data
module][trulens_eval.database.migrations.data].

## Upgrading to the latest schema revision

```python
from trulens import Tru

tru = Tru(
   database_url="<sqlalchemy_url>",
   database_prefix="trulens_" # default, may be ommitted
)
tru.migrate_database()
```

## Changing database prefix

Since `0.28.0`, all tables used by _TruLens-Eval_ are prefixed with "trulens_"
including the special `alembic_version` table used for tracking schema changes.
Upgrading to `0.28.0` for the first time will require a migration as specified
above. This migration assumes that the prefix in the existing database was
blank.

If you need to change this prefix after migration, you may need to specify the
old prefix when invoking
[migrate_database][trulens_eval.tru.Tru.migrate_database]:

```python
tru = Tru(
   database_url="<sqlalchemy_url>",
   database_prefix="new_prefix"
)
tru.migrate_database(prior_prefix="old_prefix")
```

## Copying a database

Have a look at the help text for `copy_database` and take into account all the
items under the section `Important considerations`:

```python
from trulens.database.utils import copy_database

help(copy_database)
```

Copy all data from the source database into an EMPTY target database:

```python
from trulens.database.utils import copy_database

copy_database(
    src_url="<source_db_url>",
    tgt_url="<target_db_url>",
    src_prefix="<source_db_prefix>",
    tgt_prefix="<target_db_prefix>"
)
```

::: trulens_eval.tru.Tru.migrate_database

::: trulens_eval.database.utils.copy_database

::: trulens_eval.database.migrations.data
