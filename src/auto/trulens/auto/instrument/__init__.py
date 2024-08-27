# ruff: noqa: E402, F822
"""TruLens recorders and configuration."""

from typing import TYPE_CHECKING

from trulens.auto._utils import auto as auto_utils

if TYPE_CHECKING:
    # Needed for static tools to resolve submodules:
    from trulens.core.app.base import App
    from trulens.core.app.basic import TruBasicApp
    from trulens.core.app.custom import TruCustomApp
    from trulens.core.app.virtual import TruVirtual
    from trulens.core.schema.app import AppDefinition
    from trulens.core.schema.feedback import FeedbackMode
    from trulens.instrument.langchain.tru_chain import TruChain
    from trulens.instrument.llamaindex.tru_llama import TruLlama
    from trulens.instrument.nemo.tru_rails import TruRails

_RECORDERS = {
    "TruBasicApp": ("trulens-core", "trulens.core.app.basic"),
    "TruCustomApp": ("trulens-core", "trulens.core.app.custom"),
    "TruVirtual": ("trulens-core", "trulens.core.app.virtual"),
    "TruChain": (
        "trulens-instrument-langchain",
        "trulens.instrument.langchain.tru_chain",
    ),
    "TruLlama": (
        "trulens-instrument-llamaindex",
        "trulens.instrument.llamaindex.tru_llama",
    ),
    "TruRails": (
        "trulens-instrument-nemo",
        "trulens.instrument.nemo.tru_rails",
    ),
}

_CONFIGS = {
    "FeedbackMode": ("trulens-core", "trulens.core.schema.feedback"),
}

_INTERFACES = {
    "App": ("trulens-core", "trulens.core.app.base"),
    "AppDefinition": ("trulens-core", "trulens.core.schema.app"),
}

_KINDS = {
    "recorder": _RECORDERS,
    "config": _CONFIGS,
    "interface": _INTERFACES,
}

__getattr__ = auto_utils.make_getattr_override(
    doc="TruLens app recorders.", kinds=_KINDS
)

__all__ = [
    # recorders
    "TruBasicApp",
    "TruCustomApp",
    "TruVirtual",
    "TruChain",
    "TruLlama",
    "TruRails",
    # recorder configuration
    "FeedbackMode",
    # interfaces
    "App",
    "AppDefinition",
]
