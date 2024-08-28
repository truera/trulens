# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.instruments` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.instruments import AddInstruments
from trulens.core.instruments import Instrument
from trulens.core.instruments import WithInstrumentCallbacks
from trulens.core.instruments import class_filter_disjunction
from trulens.core.instruments import class_filter_matches
from trulens.core.instruments import instrument
