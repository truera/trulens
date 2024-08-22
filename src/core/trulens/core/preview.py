from collections import defaultdict
from enum import Enum
import functools
from types import FrameType
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Set,
    TypeVar,
    Union,
)

from trulens.core.utils import python as python_utils

T = TypeVar("T")


class Feature(str, Enum):
    """Experimental feature flags.

    Use [Tru.enable_feature][trulens.core.tru.Tru.enable_feature] to enable
    these features.
    """

    OTEL_TRACING = "otel_tracing"
    """OTEL-like tracing.

    !!! Warning
        This changes how wrapped functions are processed. This setting cannot be
        changed after any wrapper is produced.
    """


class Setting(Generic[T]):
    """A setting that attains some value and can be locked from changing."""

    def __init__(self, default: T):
        self.value: T = default
        """The stored value."""

        self.locked_by: Set[FrameType] = set()
        """Set of frames (not in trulens) that have locked this value.

        If empty, it has not been locked and can be changed.
        """

    @property
    def is_locked(self) -> bool:
        """Determine if the setting is locked."""

        return len(self.locked_by) > 0

    def set(self, value: Optional[T] = None, lock: bool = False) -> T:
        """Set/Get the value.

        Set the value first if a value is provided. Lock if lock is set.

        Raises:
            ValueError: If the setting has already been locked and the value is
                different from the current value.
        """

        if value is not None:
            if len(self.locked_by) > 0 and value != self.value:
                locked_frames = ""
                for frame in self.locked_by:
                    locked_frames += (
                        f"  {python_utils.code_line(frame, show_source=True)}\n"
                    )

                raise ValueError(
                    f"Feature flag has already been set to {self.value} and cannot be changed. It has been set by:\n{locked_frames}"
                )

            self.value = value

        if lock:
            self.locked_by.add(python_utils.external_caller_frame(offset=1))

        return self.value

    def get(self, lock: bool = False) -> T:
        """Get the value of this setting.

        If lock is set, lock the setting.
        """

        return self.set(lock=lock)

    def lock(self, value: Optional[T] = None) -> T:
        """Lock the value of this setting.

        If a value is provided, attempt to set the setting first to that value.

        Raises:
            ValueError: If the setting has already been locked and the value is
                different from the current value.
        """

        return self.set(value, lock=True)


class Preview:
    """A collection of settings meant to represent preview features."""

    def __init__(self):
        self.settings: Dict[Feature, Setting[bool]] = defaultdict(
            lambda: Setting(default=False)
        )
        """The settings for the preview features."""

    def is_locked(self, flag: Union[Feature, str]) -> bool:
        """Determine if the given setting is locked."""

        return self.settings[Feature(flag)].is_locked

    def set(
        self,
        flag: Union[Feature, str],
        *,
        value: Optional[bool] = None,
        lock: bool = False,
    ) -> bool:
        """Get/Set the given feature flag to the given value.

        Sets the flag to the given value if the value parameter is set.

        Locks the flag if the lock parameter is set to True.

        Raises:
            ValueError: If the flag was already locked to a different value.
        """

        return self.settings[Feature(flag)].set(value=value, lock=lock)

    def lock(
        self, flag: Union[Feature, str], *, value: Optional[bool] = None
    ) -> bool:
        """ "Lock the given feature flag to the given value.

        If the value is not provided, lock the flag to its current value.

        Raises:
            ValueError: If the flag has already been locked to a different value.
        """

        return self.set(flag, value=value, lock=True)

    def get(self, flag: Union[str, Feature], *, lock: bool = False) -> bool:
        """Determine the value of the given feature flag by checking both global
        and instance flags.

        Instance value takes precedence over the global value.
        """

        return self.set(flag, lock=lock)

    def enable(self, flag: Union[Feature, str], *, lock: bool = False) -> bool:
        """Enable the given feature flag.

        Locks the flag if the lock parameter is set to True.

        Raises:
            ValueError: If the flag was already locked to disabled.
        """

        return self.set(flag, value=True, lock=lock)

    def disable(self, flag: Union[Feature, str], *, lock: bool = False) -> bool:
        """Disable the given feature flag.

        Locks the flag if the lock parameter is set to True.

        Raises:
            ValueError: If the flag was already locked to enabled.
        """

        return self.set(flag, value=False, lock=lock)

    def set_multiple(
        self,
        flags: Union[
            Iterable[Union[str, Feature]],
            Mapping[Union[str, Feature], bool],
        ],
        lock: bool = False,
    ):
        """Set multiple feature flags.

        If lock is set, lock the flags.

        If a dictionary is passed, the keys are the feature flags and the values
        are the values to set them to. If a list is passed, the flags are set to
        True.

        Raises:
            ValueError: If any of the flags are already locked to a different
            value than specified.
        """

        if isinstance(flags, dict):
            for flag, val in flags.items():
                self.set(flag, value=val, lock=lock)
        else:
            for flag in flags:
                self.set(flag, value=True, lock=lock)


def preview_value(
    flag: Feature, enabled: T, disabled: T, lock: bool = False
) -> T:
    """Select between two values based on the status of a feature flag.

    Locks the flag if the lock parameter is set to True.
    """

    # Here to avoid circular imports.
    from trulens.core import tru as mod_tru

    if mod_tru.Tru().feature(flag, lock=lock):
        return enabled

    return disabled


def preview_method(
    flag: Feature, enabled: Callable, disabled: Callable, lock: bool = False
) -> Callable:
    """Select between two methods based on the status of a feature flag.

    The selection happens after the method is called.

    Locks the flag if the lock parameter is set to True.
    """

    # Here to avoid circular imports.
    from trulens.core import tru as mod_tru

    @functools.wraps(enabled)
    def wrapper(*args, **kwargs):
        if mod_tru.Tru().feature(flag, lock=lock):
            return enabled(*args, **kwargs)

        return disabled(*args, **kwargs)

    return wrapper
