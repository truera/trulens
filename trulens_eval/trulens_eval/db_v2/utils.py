import inspect
import logging
from typing import Optional, List, Callable

logger = logging.getLogger(__name__)


def for_all_methods(decorator, _except: Optional[List[str]] = None):
    """Applies decorator to all methods except classmethods,
    private methods and the ones specified with `_except`"""
    def decorate(cls):
        for attr_name, attr in cls.__dict__.items():  # does not include classmethods
            if not inspect.isfunction(attr):
                continue  # skips non-method attributes
            if attr_name.startswith("_"):
                continue  # skips private methods
            if _except is not None and attr_name in _except:
                continue
            logger.debug(f"Decorating {attr_name}")
            setattr(cls, attr_name, decorator(attr))
        return cls

    return decorate


def run_before(callback: Callable):
    """Create decorator to run the callback before the function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            callback(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator
