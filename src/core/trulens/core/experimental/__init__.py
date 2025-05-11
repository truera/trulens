from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from enum import Enum
import functools
import importlib
from typing import (
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

import pydantic
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.utils import imports as import_utils
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

    SIS_COMPATIBILITY = "sis_compatibility"
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


_FEATURE_SETUPS: Dict[Feature, str] = {
    Feature.OTEL_TRACING: "trulens.experimental.otel_tracing._feature"
}
"""Mapping from experimental flags to their setup class module by name (module
containing _FeatureSetup class).

Using name here as we don't want to import them until they are needed and also
importing them here would result in a circular import.

This is used to check if the optional imports are available before enabling the
feature flags.
"""


class _FeatureSetup(pydantic.BaseModel):
    """Abstract class for utilities that manage experimental features."""

    FEATURE: ClassVar[Feature]
    """The feature flag enabling this feature."""

    REQUIREMENT: ClassVar[import_utils.ImportErrorMessages]
    """The optional imports required to use the feature."""

    @staticmethod
    @abstractmethod
    def assert_optionals_installed() -> None:
        """Assert that the optional requirements for the feature are installed."""

    @staticmethod
    @abstractmethod
    def are_optionals_installed() -> bool:
        """Check if the optional requirements for the feature are installed."""

    @staticmethod
    def assert_can_enable(feature: Feature) -> None:
        """Asserts that the given feature can be enabled.

        This is used to check if the optional imports are available before
        enabling the feature flags.
        """

        if (modname := _FEATURE_SETUPS.get(feature)) is None:
            return

        return _FeatureSetup.load_setup(modname).assert_optionals_installed()

    @staticmethod
    def can_enable(feature: Feature) -> bool:
        """Check if the given feature can be enabled.

        This is used to check if the optional imports are available before
        enabling the feature flags.
        """

        if (modname := _FEATURE_SETUPS.get(feature)) is None:
            return True

        return _FeatureSetup.load_setup(modname).are_optionals_installed()

    @staticmethod
    def load_setup(modname: str) -> Type[_FeatureSetup]:
        """Load the setup class for the given module."""

        mod = importlib.import_module(modname)

        if not hasattr(mod, "_FeatureSetup"):
            raise ImportError(
                f"Module {mod} does not contain a _FeatureSetup class."
            )

        return getattr(mod, "_FeatureSetup")


class _Setting(Generic[T]):
    """A setting that attains some value and can be prevented from further
    changes ("frozen")."""

    def __init__(self, default: T):
        self.value: T = default
        """The stored value."""

        self.frozen_by: Set[str] = set()
        """Set of representations of frames (not in trulens) that have frozen this value.

        If empty, it has not been frozen and can be changed.
        """

    @property
    def is_frozen(self) -> bool:
        """Determine if the setting is frozen."""

        return len(self.frozen_by) > 0

    def set(self, value: Optional[T] = None, freeze: bool = False) -> T:
        """Set/Get the value.

        Set the value first if a value is provided. Make it unchangeable if
        freeze is set.

        Raises:
            ValueError: If the setting has already been frozen and the value is
                different from the current value.
        """

        if value is not None:
            if len(self.frozen_by) > 0 and value != self.value:
                freezing_frames = ""
                for frame_str in self.frozen_by:
                    freezing_frames += f"  {frame_str}\n"

                raise ValueError(
                    f"Feature flag has already been set to {self.value} and cannot be changed. "
                    f"It has been frozen here:\n{freezing_frames}"
                )

            self.value = value

        if freeze:
            frame = python_utils.external_caller_frame(offset=1)
            frame_str = python_utils.code_line(frame, show_source=True)
            # Store representation instead of frame to avoid keeping references
            # that would prevent GC.
            self.frozen_by.add(frame_str)

        return self.value

    def get(self, freeze: bool = False) -> T:
        """Get the value of this setting.

        If freeze is True, freeze the setting so it cannot be changed.
        """

        return self.set(freeze=freeze)

    def freeze(self, value: Optional[T] = None) -> T:
        """Lock the value of this setting.

        If a value is provided, attempt to set the setting first to that value.

        Raises:
            ValueError: If the setting has already been frozen and the value is
                different from the current value.
        """

        return self.set(value, freeze=True)


class _Settings:
    """A collection of settings to enable/disable experimental features.

    A feature can be enabled/disabled and/or frozen so that once it is set, it
    cannot be changed again. Locking is necessary for some features like OTEL
    tracing as once components have been instrumented with old or new tracing,
    the instrumentation cannot be changed.
    """

    def __init__(self):
        self.settings: Dict[Feature, _Setting[bool]] = defaultdict(
            lambda: _Setting(default=False)
        )
        """The settings for the experimental features."""

    def is_frozen(self, flag: Union[Feature, str]) -> bool:
        """Determine if the given setting is frozen."""

        return self.settings[Feature(flag)].is_frozen

    def set(
        self,
        flag: Union[Feature, str],
        *,
        value: Optional[bool] = None,
        freeze: bool = False,
    ) -> bool:
        """Get/Set the given feature flag to the given value.

        Sets the flag to the given value if the value parameter is set. Freezes
        the flag if the freeze parameter is set to True.

        Raises:
            ValueError: If the flag was already frozen to a different value.
        """

        return self.settings[Feature(flag)].set(value=value, freeze=freeze)

    def freeze(
        self, flag: Union[Feature, str], *, value: Optional[bool] = None
    ) -> bool:
        """Lock the given feature flag to the given value.

        If the value is not provided, freeze the flag to its current value.

        Raises:
            ValueError: If the flag has already been frozen to a different value.
        """

        return self.set(flag, value=value, freeze=True)

    def get(self, flag: Union[str, Feature], *, freeze: bool = False) -> bool:
        """Determine the value of the given feature flag."""

        return self.set(flag, freeze=freeze)

    def enable(
        self, flag: Union[Feature, str], *, freeze: bool = False
    ) -> bool:
        """Enable the given feature flag.

        Freeze the flag if the freeze parameter is set to True.

        Raises:
            ValueError: If the flag was already frozen to disabled.
        """

        return self.set(flag, value=True, freeze=freeze)

    def disable(
        self, flag: Union[Feature, str], *, freeze: bool = False
    ) -> bool:
        """Disable the given feature flag.

        Freezes the flag if the freeze parameter is set to True.

        Raises:
            ValueError: If the flag was already frozen to enabled.
        """

        return self.set(flag, value=False, freeze=freeze)

    def set_multiple(
        self,
        flags: Union[
            Iterable[Union[str, Feature]],
            Mapping[Union[str, Feature], bool],
        ],
        freeze: bool = False,
    ):
        """Set multiple feature flags.

        If freeze is set, freeze the flags. If a dictionary is passed, the keys
        are the feature flags and the values are the values to set them to. If a
        list is passed, the flags are set to True.

        Raises:
            ValueError: If any of the flags are already frozen to a different
                value than specified.
        """

        if isinstance(flags, dict):
            for flag, val in flags.items():
                self.set(flag, value=val, freeze=freeze)
        else:
            for flag in flags:
                self.set(flag, value=True, freeze=freeze)


class _WithExperimentalSettings(
    pydantic.BaseModel,
    text_utils.WithIdentString,
):
    """Mixin to add experimental flags and control methods.

    Prints out messages when features are enabled/disabled, frozen, and when a
    setting fails to take up due to earlier freeze.
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
        freeze: bool = False,
    ) -> bool:
        """Get and/or set the value of the given feature flag.

        Set it first if value is given. Freeze it if `freeze` is set.

        Raises:
            ValueError: If the flag is frozen to a different value.
        """

        # NOTE(piotrm): The printouts are important as we want to make sure the
        # user is aware that they are using a experimental feature.

        flag = Feature(flag)

        was_frozen = self._experimental_feature_flags.is_frozen(flag)

        if value:
            # If the feature has optional requirements, this checks that they
            # are installed and raises an ImportError if not.
            _FeatureSetup.assert_can_enable(flag)

        original_value = self._experimental_feature_flags.get(flag)
        val = self._experimental_feature_flags.set(
            flag, value=value, freeze=freeze
        )
        changed = val != original_value

        if value is not None and changed:
            if val:
                print(
                    f"{text_utils.UNICODE_CHECK} experimental {flag} enabled."
                )
            else:
                print(
                    f"{text_utils.UNICODE_STOP} experimental {flag} disabled."
                )

        if val and freeze and not was_frozen:
            print(
                f"{text_utils.UNICODE_LOCK} experimental {flag} is enabled and cannot be changed."
            )

        return val

    def _experimental_freeze_feature(self, flag: Union[str, Feature]) -> bool:
        """Get and freeze the given feature flag."""

        return self._experimental_feature(flag, freeze=True)

    def experimental_enable_feature(self, flag: Union[str, Feature]) -> bool:
        """Enable the given feature flag.

        Raises:
            ValueError: If the flag is already frozen to disabled.
        """

        return self._experimental_feature(flag, value=True)

    def experimental_disable_feature(self, flag: Union[str, Feature]) -> bool:
        """Disable the given feature flag.

        Raises:
            ValueError: If the flag is already frozen to enabled.
        """

        return self._experimental_feature(flag, value=False)

    def experimental_feature(
        self, flag: Union[str, Feature], *, freeze: bool = False
    ) -> bool:
        """Determine the value of the given feature flag.

        If `freeze` is set, the flag will be frozen to the value returned.
        """

        return self._experimental_feature(flag, freeze=freeze)

    def experimental_set_features(
        self,
        flags: Optional[
            Union[
                Iterable[Union[str, Feature]],
                Mapping[Union[str, Feature], bool],
            ]
        ],
        freeze: bool = False,
    ):
        """Set multiple feature flags.

        If `freeze` is set, the flags will be frozen to the values given.

        Raises:
            ValueError: If any flag is already frozen to a different value than
            provided.
        """
        if is_otel_tracing_enabled():
            self._experimental_feature(
                Feature.OTEL_TRACING, value=True, freeze=True
            )
        if isinstance(flags, dict):
            for flag, val in flags.items():
                self._experimental_feature(flag, value=val, freeze=freeze)
        elif flags is not None:
            for flag in flags:
                self._experimental_feature(flag, value=True, freeze=freeze)

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
        flag: Feature, enabled: T, freeze: bool = False
    ) -> T:
        """Decorator to replace the decorated method with the given one if the
        specified feature is enabled.

        Freezes the flag if the `frozen` parameter is set.

        Example:
            ```python
            class MyClass(_WithExperimentalSettings, ...):

                def my_method_experimental(self, ...): ...

                @MyClass._experimental_method_override(
                    flag=Feature.OTEL_TRACING,
                    enabled=my_method_experimental
                )
                def my_method(self, ...): ...
            ```
        """

        def wrapper(func: T) -> T:
            @functools.wraps(func)
            def wrapped(self: _WithExperimentalSettings, *args, **kwargs):
                if self.experimental_feature(flag, freeze=freeze):
                    return enabled(*args, **kwargs)

                return func(self, *args, **kwargs)

            return wrapped

        return wrapper

    @staticmethod
    def _experimental_method(
        flag: Feature,
        enabled: Callable,
        disabled: Callable,
        freeze: bool = False,
    ) -> Callable:
        """Select between two methods based on the status of a feature flag.

        The selection happens after the method is called. Freezes the flag if
        the `freeze` parameter is set.

        Example:
            ```python
            class MyClass(_WithExperimentalSettings, ...):
                ...
                def my_method_default(self, ...): ...
                def my_method_experimental(self, ...): ...
                ...
                my_method = MyClass._experimental_method(
                    flag=Feature.OTEL_TRACING,
                    enabled=my_method_experimental,
                    disabled=my_method_default
                )
                ```
        """

        @functools.wraps(enabled)  # or disabled
        def wrapper(self: _WithExperimentalSettings, *args, **kwargs):
            if self.experimental_feature(flag, freeze=freeze):
                return enabled(*args, **kwargs)

            return disabled(*args, **kwargs)

        return wrapper
