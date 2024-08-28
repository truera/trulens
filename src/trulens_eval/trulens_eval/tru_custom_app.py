# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.core.app.custom` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.app.custom import PLACEHOLDER
from trulens.core.app.custom import UNICODE_CHECK
from trulens.core.app.custom import TruCustomApp
from trulens.core.app.custom import instrument
