from enum import Enum
import functools
from typing import Callable, TypeVar

T = TypeVar("T")


class Feature(str, Enum):
    """Experimental feature flags.

    Use [Tru.enable_feature][trulens.core.tru.Tru.enable_feature] to enable
    these features.
    """

    OTEL_TRACING = "otel_tracing"


def preview_value(flag: Feature, enabled: T, disabled: T) -> T:
    """Select between two values based on the status of a feature flag."""

    # Here to avoid circular imports.
    from trulens.core import tru as mod_tru

    if mod_tru.Tru().feature(flag):
        return enabled

    return disabled


def preview_method(
    flag: Feature, enabled: Callable, disabled: Callable
) -> Callable:
    """Select between two methods based on the status of a feature flag.

    The selection happens after the method is called.
    """

    # Here to avoid circular imports.
    from trulens.core import tru as mod_tru

    @functools.wraps(enabled)
    def wrapper(*args, **kwargs):
        if mod_tru.Tru().feature(flag):
            return enabled(*args, **kwargs)

        return disabled(*args, **kwargs)

    return wrapper
