# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.database`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

# These are all submodules of prior trulens_eval.database and current
# trulens.core.database:
# from trulens.core.database import base
from trulens.core.database import exceptions
from trulens.core.database import legacy
from trulens.core.database import migrations
from trulens.core.database import orm
from trulens.core.database import sqlalchemy
from trulens.core.database import utils


# NOTE: There trulens_eval.database.__init__.py was blank so this backwards
# compatibility module is complete assuming the interfaces to the above imported
# submodules havn't changed.
