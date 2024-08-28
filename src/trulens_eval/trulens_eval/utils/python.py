# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.utils.python` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils.python import STACK
from trulens.core.utils.python import EmptyType
from trulens.core.utils.python import OpaqueWrapper
from trulens.core.utils.python import SingletonInfo
from trulens.core.utils.python import SingletonPerName
from trulens.core.utils.python import callable_name
from trulens.core.utils.python import caller_frame
from trulens.core.utils.python import caller_frameinfo
from trulens.core.utils.python import class_name
from trulens.core.utils.python import code_line
from trulens.core.utils.python import for_all_methods
from trulens.core.utils.python import get_all_local_in_call_stack
from trulens.core.utils.python import get_first_local_in_call_stack
from trulens.core.utils.python import get_task_stack
from trulens.core.utils.python import id_str
from trulens.core.utils.python import is_really_coroutinefunction
from trulens.core.utils.python import locals_except
from trulens.core.utils.python import merge_stacks
from trulens.core.utils.python import module_name
from trulens.core.utils.python import run_before
from trulens.core.utils.python import safe_hasattr
from trulens.core.utils.python import safe_issubclass
from trulens.core.utils.python import safe_signature
from trulens.core.utils.python import stack_with_tasks
from trulens.core.utils.python import task_factory_with_stack
from trulens.core.utils.python import tru_new_event_loop
from trulens.core.utils.python import wrap_awaitable
from trulens.core.utils.python import wrap_generator
