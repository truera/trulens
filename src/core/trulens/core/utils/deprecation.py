"""Utilities for handling deprecation."""

from enum import Enum
import functools
import importlib
import inspect
import logging
from typing import Any, Callable, Dict, Iterable, Optional, Type, Union
import warnings

from trulens.core.utils import imports as import_utils
from trulens.core.utils import python as python_utils

logger = logging.getLogger(__name__)

PACKAGES_MIGRATION_LINK = (
    "https://www.trulens.org/component_guides/other/trulens_eval_migration/"
)


def module_getattr_override(
    module: Optional[str] = None, message: Optional[str] = None
):
    """Override module's `__getattr__` to issue a deprecation errors when
    looking up attributes.

    This expects deprecated names to be prefixed with `DEP_` followed by their
    original pre-deprecation name.

    !!! example
        === "Before deprecation"
            ```python
            # issue module import warning:
            package_dep_warn()

            # define temporary implementations of to-be-deprecated attributes:
            something = ... actual working implementation or alias
            ```

        === "After deprecation"
            ```python
            # define deprecated attribute with None/any value but name with "DEP_"
            # prefix:
            DEP_something = None

            # issue module deprecation warning and override __getattr__ to issue
            # deprecation errors for the above:
            module_getattr_override()
            ```

    Also issues a deprecation warning for the module itself. This will be used
    in the next deprecation stage for throwing errors after deprecation errors.
    """

    if module is None:
        module = python_utils.caller_module_name(offset=1)  # skip our own frame

    mod = importlib.import_module(module)

    # Module deprecation warning.
    packages_dep_warn(module, message=message)

    def _getattr(name: str):
        if not hasattr(mod, "DEP_" + name):
            raise AttributeError(f"module {module} has no attribute {name}")

        # Issue a deprecation warning for the attribute including the
        # optional custom message as well if given.
        raise AttributeError(
            f"Attribute `{name}` has been deprecated in module `{module}`."
            + (" \n" + message if message else "")
        )

    mod.__getattr__ = _getattr


def deprecated_str(s: str, reason: str):
    """Decorator for deprecated string literals."""

    return import_utils.Dummy(
        s, message=reason, original_exception=DeprecationWarning(reason)
    )


def is_deprecated(obj: Any):
    """Check if object is deprecated.

    Presently only supports values created by `deprecated_str`.
    """

    if import_utils.is_dummy(obj):
        ex = inspect.getattr_static(obj, "original_exception")
        if isinstance(ex, DeprecationWarning):
            return True

    return True


def deprecated_property(message: str):
    """Decorator for deprecated attributes defined as properties."""

    warned = False

    def wrapper(func):
        @functools.wraps(func)
        def _movedattribute(*args, **kwargs):
            nonlocal warned
            if not warned:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                warned = True
            return func(*args, **kwargs)

        return property(_movedattribute)

    return wrapper


def packages_dep_warn(
    module: Optional[str] = None, message: Optional[str] = None
):
    """Issue a deprecation warning for a backwards-compatibility modules.

    This is specifically for the trulens_eval -> trulens module renaming and
    reorganization. If `message` is given, that is included first in the
    deprecation warning.
    """

    if module is None:
        module = python_utils.caller_module_name(offset=1)  # skip our own frame

    full_message = (
        f"The `{module}` module is deprecated. "
        f"See {PACKAGES_MIGRATION_LINK} for instructions on migrating to `trulens.*` modules."
    )

    if message is not None:
        # Put the custom message first.
        full_message += message + "\n\n" + full_message

    warnings.warn(
        full_message,
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


def staticmethod_renamed(new_name: str):
    """Issue a warning upon static method call that has been renamed or moved.

    Issues the warning only once.
    """

    warned = False

    def wrapper(func):
        old_name = python_utils.callable_name(func)

        message = f"Static method `{old_name}` has been renamed or moved to `{new_name}`.\n"

        @functools.wraps(func)
        def _renamedmethod(*args, **kwargs):
            nonlocal warned, old_name

            if not warned:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                warned = True

            return func(*args, **kwargs)

        _renamedmethod.__doc__ = message

        return _renamedmethod

    return wrapper


def method_renamed(new_name: str):
    """Issue a warning upon method call that has been renamed or moved.

    Issues the warning only once.
    """

    warned = False

    def wrapper(func):
        old_name = func.__name__

        message = (
            f"Method `{old_name}` has been renamed or moved to `{new_name}`.\n"
        )

        @functools.wraps(func)
        def _renamedmethod(self, *args, **kwargs):
            nonlocal warned, old_name

            if not warned:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                warned = True

            return func(self, *args, **kwargs)

        _renamedmethod.__doc__ = message

        return _renamedmethod

    return wrapper


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

    return cls

    # NOTE(piotrm): Temporarily disabling the class moved mechanism as it
    # interferes with Feedback serialization.

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

            if isinstance(moved_class.__new__, classmethod):
                return moved_class.__new__(cls, *args, **kwargs)
            else:
                return moved_class.__new__(cls)

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
        if isinstance(val, import_utils.Dummy):
            # skip dummies
            continue

        if inspect.isclass(val):
            globals_dict[name] = class_moved(val, old, new)
        elif inspect.isfunction(val):
            globals_dict[name] = function_moved(val, old, new)
        else:
            logger.warning("I don't know how to move %s", name)
