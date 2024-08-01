from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

import importlib

from trulens.core.utils.imports import get_package_version


_OPTIONAL_PROVIDERS = {
    "Bedrock":
    ("trulens-provider-bedrock", "trulens.providers.bedrock.provider"),
    "Cortex": ("trulens-provider-cortex", "trulens.providers.cortex.provider"),
    "Huggingface":
    ("trulens-provider-huggingface", "trulens.providers.huggingface.provider"),
    "HuggingfaceLocal": ("trulens-provider-huggingface-local",
                         "trulens.providers.huggingfacelocal.provider"),
    "Langchain":
    ("trulens-provider-langchain", "trulens.providers.langchain.provider"),
    "LiteLLM":
    ("trulens-provider-litellm", "trulens.providers.litellm.provider"),
    "OpenAI": ("trulens-provider-openai", "trulens.providers.openai.provider"),
    "AzureOpenAI": ("trulens-provider-openai", "trulens.providers.openai.provider")
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
""")

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
""") from e
        
    raise AttributeError(f"Module {__name__} has no attribute {attr}. It has:\n  {"\n  ".join(__all__)}")

__all__ = [k for k, v in _OPTIONAL_PROVIDERS.items()]
