"""Utilities for handling deprecation."""

from enum import Enum
import warnings
import inspect
from trulens.core.utils import imports as imports_utils
from typing import Optional, Iterable, Callable, Type, Dict, Any, Union
import functools


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

    Issues the warning only once.
    """

    assert hasattr(
        func, "__name__"
    ), "Cannot create dep message without function name."

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


def class_moved(cls: Type, old_location: str, new_location: str):
    """Issue a warning upon class instantioation that has been moved to a new
    location.

    Issues the warning only once.
    """

    message = (
        f"Class `{cls.__name__}` has moved:\n"
        f"\tOld import: `from {old_location} import {cls.__name__}`\n"
        f"\tNew import: `from {new_location} import {cls.__name__}`\n "
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
    old: str,
    new: str,
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
            pass
