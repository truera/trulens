"""
# 🕸✨ Database Migration

When upgrading _TruLens_, it may sometimes be required to migrate the
database to incorporate changes in existing database created from the previously
installed version. The changes to database schemas is handled by
[Alembic](https://github.com/sqlalchemy/alembic/) while some data changes are
handled by converters in [the data
module][trulens.core.database.migrations.data].

## Upgrading to the latest schema revision

```python
from trulens.core import Tru

tru = Tru(
   database_url="<sqlalchemy_url>",
   database_prefix="trulens_" # default, may be omitted
)
tru.migrate_database()
```

## Changing database prefix

Since `0.28.0`, all tables used by _TruLens_ are prefixed with "trulens_"
including the special `alembic_version` table used for tracking schema changes.
Upgrading to `0.28.0` for the first time will require a migration as specified
above. This migration assumes that the prefix in the existing database was
blank.

If you need to change this prefix after migration, you may need to specify the
old prefix when invoking
[migrate_database][trulens.core.tru.Tru.migrate_database]:

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
from trulens.core.database.utils import copy_database

help(copy_database)
```

Copy all data from the source database into an EMPTY target database:

```python
from trulens.core.database.utils import copy_database

copy_database(
    src_url="<source_db_url>",
    tgt_url="<target_db_url>",
    src_prefix="<source_db_prefix>",
    tgt_prefix="<target_db_prefix>"
)
```

"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from typing import Iterator, List, Optional

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import BaseModel
from sqlalchemy import Engine
from trulens.core.database import base as mod_db

logger = logging.getLogger(__name__)


@contextmanager
def alembic_config(
    engine: Engine, prefix: str = mod_db.DEFAULT_DATABASE_PREFIX
) -> Iterator[Config]:
    alembic_dir = os.path.dirname(os.path.abspath(__file__))
    db_url = str(engine.url).replace("%", "%%")  # Escape any '%' in db_url
    config = Config(os.path.join(alembic_dir, "alembic.ini"))
    config.set_main_option("script_location", alembic_dir)
    config.set_main_option(
        "calling_context", "PYTHON"
    )  # skips CLI-specific setup
    config.set_main_option("sqlalchemy.url", db_url)
    config.set_main_option("trulens.table_prefix", prefix)
    config.attributes["engine"] = engine

    yield config


def upgrade_db(
    engine: Engine,
    revision: str = "head",
    prefix: str = mod_db.DEFAULT_DATABASE_PREFIX,
):
    with alembic_config(engine, prefix=prefix) as config:
        command.upgrade(config, revision)


def downgrade_db(
    engine: Engine,
    revision: str = "base",
    prefix: str = mod_db.DEFAULT_DATABASE_PREFIX,
):
    with alembic_config(engine, prefix=prefix) as config:
        command.downgrade(config, revision)


def get_current_db_revision(
    engine: Engine, prefix: str = mod_db.DEFAULT_DATABASE_PREFIX
) -> Optional[str]:
    with engine.connect() as conn:
        return MigrationContext.configure(
            conn, opts=dict(version_table=prefix + "alembic_version")
        ).get_current_revision()


def get_revision_history(
    engine: Engine, prefix: str = mod_db.DEFAULT_DATABASE_PREFIX
) -> List[str]:
    """
    Return list of all revisions, from base to head.
    Warn: Branching not supported, fails if there's more than one head.
    """
    with alembic_config(engine, prefix=prefix) as config:
        scripts = ScriptDirectory.from_config(config)
        return list(
            reversed([
                rev.revision
                for rev in scripts.iterate_revisions(lower="base", upper="head")
            ])
        )


class DbRevisions(BaseModel):
    current: Optional[str]  # current revision in the database
    history: List[str]  # all past revisions, including `latest`

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({super().__str__()})"

    @property
    def latest(self) -> str:
        """Expected revision for this release"""
        return self.history[-1]

    @classmethod
    def load(
        cls, engine: Engine, prefix: str = mod_db.DEFAULT_DATABASE_PREFIX
    ) -> DbRevisions:
        return cls(
            current=get_current_db_revision(engine, prefix=prefix),
            history=get_revision_history(engine, prefix=prefix),
        )

    @property
    def in_sync(self) -> bool:
        return self.current == self.latest

    @property
    def ahead(self) -> bool:
        return self.current is not None and self.current not in self.history

    @property
    def behind(self) -> bool:
        return self.current is None or (self.current in self.history[:-1])
