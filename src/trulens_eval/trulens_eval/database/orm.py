# ruff: noqa: E402, F403, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.database.orm` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.database.orm import ORM
from trulens.core.database.orm import BaseWithTablePrefix
from trulens.core.database.orm import make_base_for_prefix
from trulens.core.database.orm import make_orm_for_prefix
from trulens.core.database.orm import new_base
from trulens.core.database.orm import new_orm
