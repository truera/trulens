from contextlib import contextmanager
import logging
import os
from typing import List, Optional

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import BaseModel
from sqlalchemy import Engine

logger = logging.getLogger(__name__)


@contextmanager
def alembic_config(engine: Engine) -> Config:
    alembic_dir = os.path.dirname(os.path.abspath(__file__))
    db_url = str(engine.url).replace("%", "%%")  # Escape any '%' in db_url
    config = Config(os.path.join(alembic_dir, "alembic.ini"))
    config.set_main_option("script_location", alembic_dir)
    config.set_main_option(
        "calling_context", "PYTHON"
    )  # skips CLI-specific setup
    config.set_main_option("sqlalchemy.url", db_url)
    config.attributes["engine"] = engine
    yield config


def upgrade_db(engine: Engine, revision: str = "head"):
    with alembic_config(engine) as config:
        command.upgrade(config, revision)


def downgrade_db(engine: Engine, revision: str = "base"):
    with alembic_config(engine) as config:
        command.downgrade(config, revision)


def get_current_db_revision(engine: Engine) -> Optional[str]:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def get_revision_history(engine: Engine) -> List[str]:
    """
    Return list of all revisions, from base to head.
    Warn: Branching not supported, fails if there's more than one head.
    """
    with alembic_config(engine) as config:
        scripts = ScriptDirectory.from_config(config)
        return list(
            reversed(
                [
                    rev.revision for rev in
                    scripts.iterate_revisions(lower="base", upper="head")
                ]
            )
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
    def load(cls, engine: Engine) -> "DbRevisions":
        return cls(
            current=get_current_db_revision(engine),
            history=get_revision_history(engine),
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
