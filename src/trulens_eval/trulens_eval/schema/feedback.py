# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.schema.feedback` or `trulens.core.schema.select` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.schema.feedback import FeedbackCall
from trulens.core.schema.feedback import FeedbackCombinations
from trulens.core.schema.feedback import FeedbackDefinition
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.feedback import FeedbackOnMissingParameters
from trulens.core.schema.feedback import FeedbackResult
from trulens.core.schema.feedback import FeedbackResultStatus
from trulens.core.schema.select import Select
