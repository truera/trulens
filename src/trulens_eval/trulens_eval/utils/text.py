# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.utils.text`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils.text import UNICODE_CHECK
from trulens.core.utils.text import UNICODE_CLOCK
from trulens.core.utils.text import UNICODE_HOURGLASS
from trulens.core.utils.text import UNICODE_LOCK
from trulens.core.utils.text import UNICODE_SQUID
from trulens.core.utils.text import UNICODE_STOP
from trulens.core.utils.text import UNICODE_YIELD
from trulens.core.utils.text import make_retab
from trulens.core.utils.text import retab
