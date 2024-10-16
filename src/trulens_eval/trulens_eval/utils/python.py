# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.core.utils.python` instead.
"""

from __future__ import annotations

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()
import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Optional,
    Type,
    TypeVar,
)

from trulens.core.utils.python import STACK
from trulens.core.utils.python import OpaqueWrapper
from trulens.core.utils.python import T
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
from trulens.core.utils.python import logger
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


@dataclasses.dataclass
class SingletonInfo(Generic[T]):
    """
    Information about a singleton instance.
    """

    val: T
    """The singleton instance."""

    cls: Type[T]
    """The class of the singleton instance."""

    frameinfo_codeline: Optional[str]
    """The frame where the singleton was created.

    This is used for showing "already created" warnings. This is intentionally
    not the frame itself but a rendering of it to avoid maintaining references
    to frames and all of the things a frame holds onto.
    """

    name: Optional[str] = None
    """The name of the singleton instance.

    This is used for the SingletonPerName mechanism to have a separate singleton
    for each unique name (and class).
    """

    def __init__(self, name: str, val: Any):
        self.val = val
        self.cls = val.__class__
        self.name = name
        self.frameinfo_codeline = code_line(
            caller_frameinfo(offset=2), show_source=True
        )

    def warning(self):
        """Issue warning that this singleton already exists."""

        logger.warning(
            (
                "Singleton instance of type %s already created at:\n%s\n"
                "You can delete the singleton by calling `<instance>.delete_singleton()` or \n"
                f"""  ```python
  from trulens.core.utils.python import SingletonPerName
  SingletonPerName.delete_singleton_by_name(name="{self.name}", cls={self.cls.__name__})
  ```
            """
            ),
            self.cls.__name__,
            self.frameinfo_codeline,
        )


class SingletonPerName:
    """
    Class for creating singleton instances except there being one instance max,
    there is one max per different `name` argument. If `name` is never given,
    reverts to normal singleton behavior.
    """

    # Hold singleton instances here.
    _instances: Dict[Hashable, SingletonInfo[SingletonPerName]] = {}

    # Need some way to look up the name of the singleton instance. Cannot attach
    # a new attribute to instance since some metaclasses don't allow this (like
    # pydantic). We instead create a map from instance address to name.
    _id_to_name_map: Dict[int, Optional[str]] = {}

    def warning(self):
        """Issue warning that this singleton already exists."""

        name = SingletonPerName._id_to_name_map[id(self)]
        k = self.__class__.__name__, name
        if k in SingletonPerName._instances:
            SingletonPerName._instances[k].warning()
        else:
            raise RuntimeError(
                f"Instance of singleton type/name {k} does not exist."
            )

    def __new__(
        cls: Type[SingletonPerName],
        *args,
        name: Optional[str] = None,
        **kwargs,
    ) -> SingletonPerName:
        """
        Create the singleton instance if it doesn't already exist and return it.
        """

        k = cls.__name__, name

        if k not in cls._instances:
            logger.debug(
                "*** Creating new %s singleton instance for name = %s ***",
                cls.__name__,
                name,
            )
            # If exception happens here, the instance should not be added to
            # _instances.
            instance = super().__new__(cls)

            SingletonPerName._id_to_name_map[id(instance)] = name
            info: SingletonInfo = SingletonInfo(name=name, val=instance)
            SingletonPerName._instances[k] = info
        else:
            info = SingletonPerName._instances[k]
        obj = info.val
        assert isinstance(obj, cls)
        return obj

    @staticmethod
    def delete_singleton_by_name(
        name: str, cls: Optional[Type[SingletonPerName]] = None
    ):
        """
        Delete the singleton instance with the given name.

        This can be used for testing to create another singleton.

        Args:
            name: The name of the singleton instance to delete.

            cls: The class of the singleton instance to delete. If not given, all
                instances with the given name are deleted.
        """
        for k, v in list(SingletonPerName._instances.items()):
            if k[1] == name:
                if cls is not None and v.cls != cls:
                    continue

                del SingletonPerName._instances[k]
                del SingletonPerName._id_to_name_map[id(v.val)]

    def delete_singleton(self):
        """
        Delete the singleton instance. Can be used for testing to create another
        singleton.
        """
        id_ = id(self)

        if id_ in SingletonPerName._id_to_name_map:
            name = SingletonPerName._id_to_name_map[id_]
            del SingletonPerName._id_to_name_map[id_]
            del SingletonPerName._instances[(self.__class__.__name__, name)]
        else:
            logger.warning("Instance %s not found in our records.", self)
