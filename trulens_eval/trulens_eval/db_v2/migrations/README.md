# Database Migrations
Database schema migrations are handled with [Alembic](https://github.com/sqlalchemy/alembic/)

## Migrating to the latest revision

```python
from trulens_eval import Tru

tru = Tru(database_url="<sqlalchemy_url>")
tru.migrate_database()
```

## Creating a new revision

1. Make sure you have an existing database at the latest schema
2. Edit the [SQLAlchemy models](../models.py)
3. Run `alembic revision --autogenerate -m "<short_description>" --rev-id "<next_integer_version>"`
4. Look at the migration plan generated at [versions](./versions) and edit if necessary
