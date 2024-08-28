# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.dashboard.notebook_utils` instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.dashboard.notebook_utils import is_notebook
from trulens.dashboard.notebook_utils import setup_widget_stdout_stderr
