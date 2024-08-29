# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.providers.litellm.endpoint` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.providers.litellm.endpoint import LiteLLMCallback
from trulens.providers.litellm.endpoint import LiteLLMEndpoint
