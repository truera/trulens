# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.dashboard.run`
    instead.
"""

import warnings

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()


def main():
    warnings.warn(
        "This main method is deprecated. Use `trulens.dashboard.run.run_dashboard` instead.",
        DeprecationWarning,
    )
    from trulens.dashboard.run import run_dashboard

    run_dashboard()
