from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

# TODO: get this from poetry
__version_info__ = (0, 33, 0)
"""Version number components for major, minor, patch."""

__version__ = ".".join(map(str, __version_info__))
"""Version number string."""

# This check is intentionally done ahead of the other imports as we want to
# print out a nice warning/error before an import error happens further down
# this sequence.
# from trulens.utils.imports import check_imports

# check_imports()

import importlib

from trulens.core import schema as core_schema
from trulens.core import tru as mod_tru
from trulens.core.app import basic as mod_tru_basic_app
from trulens.core.app import custom as mod_tru_custom_app
from trulens.core.app import virtual as mod_tru_virtual
from trulens.core.feedback import feedback as mod_feedback
from trulens.core.feedback import provider as mod_provider
from trulens.core.schema import feedback as feedback_schema
from trulens.core.utils import threading as threading_utils
from trulens.core.utils.imports import get_package_version


# Common classes:
Tru = mod_tru.Tru

TP = threading_utils.TP
Feedback = mod_feedback.Feedback
Provider = mod_provider.Provider
FeedbackMode = feedback_schema.FeedbackMode
# TODO: other enums
Select = core_schema.Select

# Optional providers:
_OPTIONAL_PROVIDERS = {
    "Bedrock": (
        "trulens-providers-bedrock",
        "trulens.providers.bedrock.provider",
    ),
    "Cortex": ("trulens-providers-cortex", "trulens.providers.cortex.provider"),
    "Huggingface": (
        "trulens-providers-huggingface",
        "trulens.providers.huggingface.provider",
    ),
    "HuggingfaceLocal": (
        "trulens-providers-huggingface-local",
        "trulens.providers.huggingfacelocal.provider",
    ),
    "Langchain": (
        "trulens-providers-langchain",
        "trulens.providers.langchain.provider",
    ),
    "LiteLLM": (
        "trulens-providers-litellm",
        "trulens.providers.litellm.provider",
    ),
    "OpenAI": ("trulens-providers-openai", "trulens.providers.openai.provider"),
    "AzureOpenAI": (
        "trulens-providers-openai",
        "trulens.providers.openai.provider",
    ),
}

# Built-in recorders:
TruBasicApp = mod_tru_basic_app.TruBasicApp
TruCustomApp = mod_tru_custom_app.TruCustomApp
TruVirtual = mod_tru_virtual.TruVirtual

# Optional recorders:
_OPTIONAL_APPS = {
    "TruChain": (
        "trulens-instrument-langchain",
        "trulens.instrument.langhain.tru_chain",
    ),
    "TruLlama": (
        "trulens-instrument-llamaindex",
        "trulens.instrument.llama.tru_llama",
    ),
    "TruRails": (
        "trulens-instrument-nemo",
        "trulens.instrument.nemo.tru_rails",
    ),
}


def __getattr__(attr):
    if attr in _OPTIONAL_PROVIDERS:
        package_name, module_name = _OPTIONAL_PROVIDERS[attr]

        installed_version = get_package_version(package_name)

        if installed_version is None:
            raise ImportError(
                f"""The {attr} provider requires the {package_name} package. You can install it with pip:
    ```bash
    pip install {package_name}
    ```
"""
            )

        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, attr)

        except ImportError as e:
            raise ImportError(
                f"""Could not import the {attr} provider. You might need to re-install {package_name}:
    ```bash
    pip uninstall -y {package_name}
    pip install {package_name}
    ```
"""
            ) from e

    elif attr in _OPTIONAL_APPS:
        package_name, module_name = _OPTIONAL_APPS[attr]

        installed_version = get_package_version(package_name)

        if installed_version is None:
            raise ImportError(
                f"""The {attr} recorder requires the {package_name} package. You can install it with pip:
    ```bash
    pip install {package_name}
    ```
"""
            )

        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, attr)

        except ImportError as e:
            raise ImportError(
                f"""Could not import the {attr} recorder. You might need to re-install {package_name}:
    ```bash
    pip uninstall -y {package_name}
    pip install {package_name}
    ```
"""
            ) from e

    raise AttributeError(
        f"Module {__name__} has no attribute {attr}. It has:\n  {"\n  ".join(__all__)}"
    )


__all__ = [
    "Tru",  # main interface
    # app types
    "TruBasicApp",
    "TruCustomApp",
    "TruVirtual",
    *list(_OPTIONAL_APPS.keys()),
    # app setup
    "FeedbackMode",
    # feedback setup
    "Feedback",
    "Select",
    # feedback providers
    "Provider",
    *list(_OPTIONAL_PROVIDERS.keys()),
    # misc utility
    "TP",
]
