# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.utils.containers`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils.containers import BlockingSet
from trulens.core.utils.containers import dict_merge_with
from trulens.core.utils.containers import dict_set_with
from trulens.core.utils.containers import dict_set_with_multikey
from trulens.core.utils.containers import first
from trulens.core.utils.containers import is_empty
from trulens.core.utils.containers import iterable_peek
from trulens.core.utils.containers import second
from trulens.core.utils.containers import third
