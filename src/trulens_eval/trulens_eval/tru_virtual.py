# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.core.app.virtual` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.app.virtual import TruVirtual
from trulens.core.app.virtual import VirtualApp
from trulens.core.app.virtual import VirtualRecord
