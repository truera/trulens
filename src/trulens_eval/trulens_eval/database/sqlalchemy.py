# ruff: noqa: E402, F403, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.database.sqlalchemy` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.database.sqlalchemy import AppsExtractor
from trulens.core.database.sqlalchemy import SnowflakeImpl
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.database.sqlalchemy import _make_no_perf
from trulens.core.database.sqlalchemy import flatten

no_perf = _make_no_perf()
