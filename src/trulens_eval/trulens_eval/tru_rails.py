# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
     `trulens.apps.nemo.tru_rails` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.apps.nemo.tru_rails import FeedbackActions
from trulens.apps.nemo.tru_rails import RailsActionSelect
from trulens.apps.nemo.tru_rails import RailsInstrument
from trulens.apps.nemo.tru_rails import TruRails
from trulens.apps.nemo.tru_rails import registered_feedback_functions
