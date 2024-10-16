# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.feedback.endpoint` or `trulens.feedback.dummy.endpoint`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.feedback.endpoint import DEFAULT_RPM
from trulens.core.feedback.endpoint import INSTRUMENT
from trulens.core.feedback.endpoint import Endpoint
from trulens.core.feedback.endpoint import EndpointCallback
from trulens.core.utils.threading import DEFAULT_NETWORK_TIMEOUT
from trulens.feedback.dummy.endpoint import DummyEndpoint
