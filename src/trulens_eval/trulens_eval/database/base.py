# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.database.base`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.database.base import DB
from trulens.core.database.base import DEFAULT_DATABASE_FILE
from trulens.core.database.base import DEFAULT_DATABASE_PREFIX
from trulens.core.database.base import DEFAULT_DATABASE_REDACT_KEYS
from trulens.core.database.base import MULTI_CALL_NAME_DELIMITER
