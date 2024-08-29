# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.apps.basic` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.apps.basic import TruBasicApp
from trulens.apps.basic import TruBasicCallableInstrument
from trulens.apps.basic import TruWrapperApp
