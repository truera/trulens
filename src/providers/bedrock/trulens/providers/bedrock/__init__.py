"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-providers-bedrock` package installed.

    ```bash
    pip install trulens-providers-bedrock
    ```

[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that makes
FMs from leading AI startups and Amazon available via an API, so you can choose
from a wide range of FMs to find the model that is best suited for your use case

All feedback functions listed in the base [LLMProvider
class][trulens.feedback.LLMProvider] can be run with AWS
Bedrock.
"""

from importlib.metadata import version

from trulens.core.utils.imports import safe_importlib_package_name
from trulens.providers.bedrock.provider import Bedrock

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "Bedrock",
]
