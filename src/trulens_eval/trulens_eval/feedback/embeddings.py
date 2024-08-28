# ruff: noqa: E402, F403, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.feedback.embeddings` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.feedback.embeddings import Embeddings
