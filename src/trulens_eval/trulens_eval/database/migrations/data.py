# ruff: noqa: E402, F403, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.database.migrations.data` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.database.migrations.data import data_migrate
from trulens.core.database.migrations.data import sql_alchemy_migration_versions
from trulens.core.database.migrations.data import sqlalchemy_upgrade_paths
