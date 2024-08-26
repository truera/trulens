# ruff: noqa: E402, F822

from typing import TYPE_CHECKING

from trulens.auto._utils import auto as auto_utils

if TYPE_CHECKING:
    # Needed for static tools to resolve submodules:
    from trulens.core.app.basic import TruBasicApp
    from trulens.core.app.custom import TruCustomApp
    from trulens.core.app.virtual import TruVirtual
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

_KINDS = {
    "recorder": _RECORDERS,
}

help, help_str = auto_utils.make_help_str(_KINDS)

__getattr__ = auto_utils.make_getattr_override(_KINDS, help_str=help_str)

__all__ = [
    "TruBasicApp",
    "TruCustomApp",
    "TruVirtual",
    "TruChain",
    "TruLlama",
    "TruRails",
]
