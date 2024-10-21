from collections import defaultdict
from enum import Enum
import functools
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

import pydantic
from trulens.core.utils import python as python_utils
from trulens.core.utils import text as text_utils

T = TypeVar("T")


class Feature(str, Enum):
    """Experimental feature flags.

    Use [TruSession.experimental_enable_feature][trulens.core.session.TruSession.experimental_enable_feature] to enable
    these features:

    Examples:
        ```python
        from trulens.core.session import TruSession
        from trulens.core.experimental import Feature

        session = TruSession()

        session.experimental_enable_feature(Feature.OTEL_TRACING)
        ```
    """

    OTEL_TRACING = "otel_tracing"
    """OTEL-like tracing.

    !!! Warning
        This changes how wrapped functions are processed. This setting cannot be
        changed after any wrapper is produced.
    """

    @classmethod
    def _repr_all(cls) -> str:
        """Return a string representation of all the feature flags."""

        ret = ""
        for flag in cls:
            ret += f'  {flag.name} = "{flag.value}"\n'

        return ret

    @classmethod
    def _missing_(cls, value: str):
        raise ValueError(
            f"Invalid feature flag `{value}`. Available flags are:\n{cls._repr_all()}"
        )


class _Setting(Generic[T]):
    """A setting that attains some value and can be locked from changing."""

    def __init__(self, default: T):
        self.value: T = default
        """The stored value."""

        self.locked_by: Set[str] = set()
        """Set of representations of frames (not in trulens) that have locked this value.

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
                for frame_str in self.locked_by:
                    locked_frames += f"  {frame_str}\n"

                raise ValueError(
                    f"Feature flag has already been set to {self.value} and cannot be changed. "
                    f"It has been locked here:\n{locked_frames}"
                )

            self.value = value

        if lock:
            frame = python_utils.external_caller_frame(offset=1)
            frame_str = python_utils.code_line(frame, show_source=True)
            # Store representation instead of frame to avoid keeping references
            # that would prevent GC.
            self.locked_by.add(frame_str)

        return self.value

    def get(self, lock: bool = False) -> T:
        """Get the value of this setting.

        If lock is True, lock the setting.
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


class _Settings:
    """A collection of settings to enable/disable experimental features.

    A feature can be enabled/disabled and/or locked so that once it is set, it
    cannot be changed again. Locking is necessary for some features like OTEL
    tracing as once components have been instrumented with old or new tracing,
    the instrumentation cannot be changed.
    """

    def __init__(self):
        self.settings: Dict[Feature, _Setting[bool]] = defaultdict(
            lambda: _Setting(default=False)
        )
        """The settings for the experimental features."""

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

        Sets the flag to the given value if the value parameter is set. Locks
        the flag if the lock parameter is set to True.

        Raises:
            ValueError: If the flag was already locked to a different value.
        """

        return self.settings[Feature(flag)].set(value=value, lock=lock)

    def lock(
        self, flag: Union[Feature, str], *, value: Optional[bool] = None
    ) -> bool:
        """Lock the given feature flag to the given value.

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

        If lock is set, lock the flags. If a dictionary is passed, the keys are
        the feature flags and the values are the values to set them to. If a
        list is passed, the flags are set to True.

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


class _WithExperimentalSettings(
    pydantic.BaseModel,
    text_utils.WithIdentString,
):
    """Mixin to add experimental flags and control methods.

    Prints out messages when features are enabled/disabled locked and when
    a setting fails to take up due to locking.
    """

    _experimental_feature_flags: _Settings = pydantic.PrivateAttr(
        default_factory=_Settings
    )
    """EXPERIMENTAL: Flags to control experimental features."""

    def _experimental_feature(
        self,
        flag: Union[str, Feature],
        *,
        value: Optional[bool] = None,
        lock: bool = False,
    ) -> bool:
        """Get and/or set the value of the given feature flag.

        Set it first if value is given. Lock it if lock is set.

        Raises:
            ValueError: If the flag is locked to a different value.
        """

        # NOTE(piotrm): The printouts are important as we want to make sure the
        # user is aware that they are using a experimental feature.

        flag = Feature(flag)

        was_locked = self._experimental_feature_flags.is_locked(flag)

        original_value = self._experimental_feature_flags.get(flag)
        val = self._experimental_feature_flags.set(flag, value=value, lock=lock)
        changed = val != original_value

        if value is not None and changed:
            if val:
                print(
                    f"{text_utils.UNICODE_CHECK} experimental {flag} enabled for {self._ident_str()}."
                )
            else:
                print(
                    f"{text_utils.UNICODE_STOP} experimental {flag} disabled for {self._ident_str()}"
                )

        if val and lock and not was_locked:
            print(
                f"{text_utils.UNICODE_LOCK} experimental {flag} is enabled and cannot be changed."
            )

        return val

    def _experimental_lock_feature(self, flag: Union[str, Feature]) -> bool:
        """Get and lock the given feature flag."""

        return self._experimental_feature(flag, lock=True)

    def experimental_enable_feature(self, flag: Union[str, Feature]) -> bool:
        """Enable the given feature flag.

        Raises:
            ValueError: If the flag is already locked to disabled.
        """

        return self._experimental_feature(flag, value=True)

    def experimental_disable_feature(self, flag: Union[str, Feature]) -> bool:
        """Disable the given feature flag.

        Raises:
            ValueError: If the flag is already locked to enabled.
        """

        return self._experimental_feature(flag, value=False)

    def experimental_feature(
        self, flag: Union[str, Feature], *, lock: bool = False
    ) -> bool:
        """Determine the value of the given feature flag.

        If lock is set, the flag will be locked to the value returned.
        """

        return self._experimental_feature(flag, lock=lock)

    def experimental_set_features(
        self,
        flags: Union[
            Iterable[Union[str, Feature]],
            Mapping[Union[str, Feature], bool],
        ],
        lock: bool = False,
    ):
        """Set multiple feature flags.

        If lock is set, the flags will be locked to the values given.

        Raises:
            ValueError: If any flag is already locked to a different value than
            provided.
        """

        if isinstance(flags, dict):
            for flag, val in flags.items():
                self._experimental_feature(flag, value=val, lock=lock)
        else:
            for flag in flags:
                self._experimental_feature(flag, value=True, lock=lock)

    def _experimental_assert_feature(
        self, flag: Feature, purpose: Optional[str] = None
    ):
        """Raise a ValueError if the given feature flag is not enabled.

        Gives instructions on how to enable the feature flag if error gets
        raised."""

        flag = Feature(flag)

        if purpose is None:
            purpose = "."
        else:
            purpose = f" for {purpose}."

        if not self.experimental_feature(flag):
            raise ValueError(
                f"""Feature flag {flag} is not enabled{purpose} You can enable it in two ways:
    ```python
    from trulens.core.experimental import Feature

    # Enable for this instance when creating it:
    val = {self.__class__.__name__}(experimental_feature_flags=[{flag}]

    # Enable for this instance after it has been created (for features that allows this):
    val.experimental_enable_feature({flag})
    ```
"""
            )

    @staticmethod
    def _experimental_method_override(
        flag: Feature, enabled: T, lock: bool = False
    ) -> T:
        """Decorator to replace the decorated method with the given one if the
        specified feature is enabled.

        Locks the flag if the lock parameter is set to True.

        Example:
            ```python
            class MyClass(WithExperimentalSettings, ...):

                def my_method_experimental(self, ...): ...

                @MyClass.experimental_method_override(
                    flag=Feature.OTEL_TRACING,
                    enabled=my_method_experimental
                )
                def my_method(self, ...): ...
            ```
        """

        def wrapper(func: T) -> T:
            @functools.wraps(func)
            def wrapped(self: _WithExperimentalSettings, *args, **kwargs):
                if self.experimental_feature(flag, lock=lock):
                    return enabled(*args, **kwargs)

                return func(self, *args, **kwargs)

            return wrapped

        return wrapper

    @staticmethod
    def _experimental_method(
        flag: Feature, enabled: Callable, disabled: Callable, lock: bool = False
    ) -> Callable:
        """Select between two methods based on the status of a feature flag.

        The selection happens after the method is called. Locks the flag if the lock
        parameter is set to True.

        Example:
            ```python
            class MyClass(WithExperimentalSettings, ...):
                ...
                def my_method_default(self, ...): ...
                def my_method_experimental(self, ...): ...
                ...
                my_method = MyClass.experimental_method(
                    flag=Feature.OTEL_TRACING,
                    enabled=my_method_experimental,
                    disabled=my_method_default
                )
                ```
        """

        @functools.wraps(enabled)  # or disabled
        def wrapper(self: _WithExperimentalSettings, *args, **kwargs):
            if self.experimental_feature(flag, lock=lock):
                return enabled(*args, **kwargs)

            return disabled(*args, **kwargs)

        return wrapper
