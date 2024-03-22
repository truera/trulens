# 🗄🕸 -> 🗄✨ Database Migration

Database schema revisions are handled with
[Alembic](https://github.com/sqlalchemy/alembic/)

## Upgrading to the latest schema revision

```python
from trulens_eval import Tru

tru = Tru(
   database_url="<sqlalchemy_url>",
   database_prefix="trulens_"
)
tru.migrate_database()
# If database contains more than one alembic-versioned app, the prior prefix used by trulens may be required:
# tru.migrate_database(prior_prefix="something")
```

::: trulens_eval.tru.Tru.migrate_database

## Copying a database

Have a look at the help text for `copy_database` and take into account all the
items under the section `Important considerations`:

```python
from trulens_eval.database.utils import copy_database

help(copy_database)
```

Copy all data from the source database into an EMPTY target database:

```python
from trulens_eval.database.utils import copy_database

copy_database(
    src_url="<source_db_url>",
    tgt_url="<target_db_url>",
    src_prefix="<source_db_prefix>",
    tgt_prefix="<target_db_prefix>"
)
```

::: trulens_eval.database.utils.copy_database
