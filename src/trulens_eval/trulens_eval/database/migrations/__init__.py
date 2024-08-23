# ruff: noqa: E402, F403, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.database.migrations` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

# NOTE: This file had contents in trulens_eval but none were public or aliases.
# Because of that, this backwards compatibility module is empty.

# TODO: Get the non-exported names here too.

from trulens.core.database.migrations import DbRevisions
from trulens.core.database.migrations import alembic_config
from trulens.core.database.migrations import downgrade_db
from trulens.core.database.migrations import get_current_db_revision
from trulens.core.database.migrations import get_revision_history
from trulens.core.database.migrations import upgrade_db
