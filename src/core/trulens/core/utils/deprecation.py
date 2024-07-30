"""Utilities for handling deprecation."""

import warnings
import inspect
from trulens.core.utils import imports as imports_utils
from typing import Optional, Iterable

def class_moved(cls, old_location: str, new_location: str):
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

    class _MovedClass(moved_class):
        def __new__(cls, *args, **kwargs):
            nonlocal warned

            if not warned:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                warned = True

            return moved_class.__new__(moved_class, *args, **kwargs)

    _MovedClass.__doc__ = message

    return _MovedClass

def moved(globals_dict, old: str, new: str, names: Optional[Iterable[str]] = None):
    """Replace all classes in the given dictionary with ones that issue a
    deprecation warning upon initialization.
    
    You can use this with module `globals_dict=globals()` and `names=__all__` to
    deprecate all exposed module members. 

    Args:
        globals_dict: The dictionary to update. See [globals][globals].

        old: The old location of the classes.

        new: The new location of the classes.

        names: The names of the classes to update. If None, all classes are updated.
    """

    if names is None:
        names = globals_dict.keys()

    for name in names:
        val = globals_dict[name]
        if not inspect.isclass(val):
            # Don't currently have a good way to make aliases for non classes.
            continue

        cls = globals_dict[name]
        if not isinstance(cls, imports_utils.Dummy):
            globals_dict[name] = class_moved(cls, old, new)
