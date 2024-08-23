"""Utilities for handling deprecation."""

from enum import Enum
import functools
import inspect
import logging
from typing import Any, Callable, Dict, Iterable, Optional, Type, Union
import warnings

from trulens.core.utils import imports as imports_utils

logger = logging.getLogger(__name__)

PACKAGES_MIGRATION_LINK = (
    "https://trulens.org/docs/migration-guide"  # TODO: update link
)


def packages_dep_warn(module: str):
    """Issue a deprecation warning for a backwards-compatibility modules.

    This is specifically for the trulens_eval -> trulens module renaming and
    reorganization.
    """

    warnings.warn(
        f"The `{module}` module is deprecated. "
        f"See {PACKAGES_MIGRATION_LINK} for instructions on migrating to `trulens.*` modules.",
        DeprecationWarning,
        stacklevel=3,
    )


def has_deprecated(obj: Union[Callable, Type]) -> bool:
    """Check if a function or class has been deprecated."""

    return has_moved(obj)


def has_moved(obj: Union[Callable, Type]) -> bool:
    """Check if a function or class has been moved."""

    return (
        hasattr(obj, "__doc__")
        and obj.__doc__ is not None
        and "has moved:\n" in obj.__doc__
    )


def function_moved(func: Callable, old: str, new: str):
    """Issue a warning upon function call that has been moved to a new location.

    Issues the warning only once. The given callable must have a name, so it
    cannot be a lambda.
    """

    if not hasattr(func, "__name__"):
        raise TypeError("Cannot create dep message without function name.")

    message = (
        f"Function `{func.__name__}` has moved:\n"
        f"\tOld import: `from {old} import {func.__name__}`\n"
        f"\tNew import: `from {new} import {func.__name__}`\n "
    )

    warned = False

    @functools.wraps(func)
    def _movedfunction(*args, **kwargs):
        nonlocal warned

        if not warned:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            warned = True

        return func(*args, **kwargs)

    _movedfunction.__doc__ = message

    return _movedfunction


def class_moved(
    cls: Type,
    old_location: Optional[str] = None,
    new_location: Optional[str] = None,
):
    """Issue a warning upon class instantiation that has been moved to a new
    location.

    Issues the warning only once.
    """

    if new_location is None:
        new_location = cls.__module__

    message = (
        f"Class `{cls.__name__}` has moved:\n"
        + (
            f"\tOld import: `from {old_location} import {cls.__name__}`\n"
            if old_location
            else ""
        )
        + f"\tNew import: `from {new_location} import {cls.__name__}`\n "
        + f"See {PACKAGES_MIGRATION_LINK} for instructions on migrating to `trulens` modules."
    )

    warned = False
    moved_class = cls

    if issubclass(cls, Enum):
        # Enums cannot be extended.
        return cls

    class _MovedClass(moved_class):
        def __new__(cls, *args, **kwargs):
            nonlocal warned

            if not warned:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                warned = True

            return moved_class.__new__(moved_class, *args, **kwargs)

    _MovedClass.__doc__ = message

    return _MovedClass


def moved(
    globals_dict: Dict[str, Any],
    old: Optional[str] = None,
    new: Optional[str] = None,
    names: Optional[Iterable[str]] = None,
):
    """Replace all classes or function in the given dictionary with ones that
    issue a deprecation warning upon initialization or invocation.

    You can use this with module `globals_dict=globals()` and `names=__all__` to
    deprecate all exposed module members.

    Args:
        globals_dict: The dictionary to update. See [globals][globals].

        old: The old location of the classes.

        new: The new location of the classes.

        names: The names of the classes or functions to update. If None, all
            classes and functions are updated.
    """

    if names is None:
        names = globals_dict.keys()

    for name in names:
        val = globals_dict[name]
        if isinstance(val, imports_utils.Dummy):
            # skip dummies
            continue

        if inspect.isclass(val):
            globals_dict[name] = class_moved(val, old, new)
        elif inspect.isfunction(val):
            globals_dict[name] = function_moved(val, old, new)
        else:
            logger.warning("I don't know how to move %s", name)
