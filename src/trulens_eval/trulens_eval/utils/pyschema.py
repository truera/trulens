# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.utils.pyschema` or `trulens.core.utils.constants` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils.constants import CIRCLE
from trulens.core.utils.pyschema import CLASS_INFO
from trulens.core.utils.pyschema import ERROR
from trulens.core.utils.pyschema import NOSERIO
from trulens.core.utils.pyschema import Bindings
from trulens.core.utils.pyschema import Class
from trulens.core.utils.pyschema import Function
from trulens.core.utils.pyschema import FunctionOrMethod
from trulens.core.utils.pyschema import Method
from trulens.core.utils.pyschema import Module
from trulens.core.utils.pyschema import Obj
from trulens.core.utils.pyschema import WithClassInfo
from trulens.core.utils.pyschema import builtin_init_dummy
from trulens.core.utils.pyschema import clean_attributes
from trulens.core.utils.pyschema import is_noserio
from trulens.core.utils.pyschema import noserio
from trulens.core.utils.pyschema import object_module
from trulens.core.utils.pyschema import safe_getattr
