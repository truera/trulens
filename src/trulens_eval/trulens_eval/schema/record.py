# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.schema.record` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.schema.record import Record
from trulens.core.schema.record import RecordAppCall
from trulens.core.schema.record import RecordAppCallMethod
