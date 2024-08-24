# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.core.app.basic` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.app.basic import TruBasicApp
from trulens.core.app.basic import TruBasicCallableInstrument
from trulens.core.app.basic import TruWrapperApp
