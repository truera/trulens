"""Aliases.

Aliases to new locations of various names. This module is deprecated and will be
removed in a future release.
"""

from trulens_eval import dep_warn
from trulens.core.utils import deprecation

# NOTE: This file had contents in trulens_eval none were public or aliases.
# Because of that, this backwards compatibility module is empty.

dep_warn("trulens_eval.database.migrations")
