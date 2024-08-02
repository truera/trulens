__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import importlib

from trulens.core.utils.imports import get_package_version

if TYPE_CHECKING:
    from trulens.providers.bedrock.provider import Bedrock
    from trulens.providers.cortex.provider import Cortex
    from trulens.providers.huggingface.provider import Huggingface
    from trulens.providers.huggingfacelocal.provider import HuggingfaceLocal
    from trulens.providers.langchain.provider import Langchain
    from trulens.providers.litellm.provider import LiteLLM
    from trulens.providers.openai.provider import AzureOpenAI, OpenAI

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

    raise AttributeError(
        f"Module {__name__} has no attribute {attr}. It has:\n  {"\n  ".join(__all__)}"
    )

# This has to be statically assigned though we would prefer to use _OPTIONAL_PROVIDERS.keys():
__all__ = [
    "Bedrock",
    "Cortex",
    "Huggingface", "HuggingfaceLocal", "Langchain", "LiteLLM", "OpenAI", "AzureOpenAI"
]
