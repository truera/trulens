# Database Migrations
Database schema revisions are handled with [Alembic](https://github.com/sqlalchemy/alembic/)

## Upgrading to the latest schema revision

```python
from trulens_eval import Tru

tru = Tru(database_url="<sqlalchemy_url>")
tru.migrate_database()
```

## Creating a new schema revision

1. Make sure you have an existing database at the latest schema
2. Edit the [SQLAlchemy models](../orm.py)
3. Run `export SQLALCHEMY_URL="<url>" && alembic revision --autogenerate -m "<short_description>" --rev-id "<next_integer_version>"`
4. Look at the migration script generated at [versions](./versions) and edit if necessary
5. Open a PR including the migration script and updated models 


## Migrating between databases
Have a look at the help text for `_copy_database` and take into
account all the items under the section `Important considerations`:

```python

from trulens_eval.database.utils import _copy_database

help(_copy_database)
```

Copy all data from the source database into an EMPTY target database:

```python

from trulens_eval.database.utils import _copy_database

_copy_database(
    src_url="<source_db_url>",
    tgt_url="<target_db_url>"
)
```
