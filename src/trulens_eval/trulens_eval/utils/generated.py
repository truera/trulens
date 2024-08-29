# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.feedback.generated` instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.feedback.generated import ParseError
from trulens.feedback.generated import re_0_10_rating
from trulens.feedback.generated import validate_rating
