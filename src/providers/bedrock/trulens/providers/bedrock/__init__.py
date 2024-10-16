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

# WARNING: This file does not follow the no-init aliases import standard.

from importlib.metadata import version

from trulens.core.utils import imports as import_utils
from trulens.providers.bedrock.provider import Bedrock

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)

__all__ = [
    "Bedrock",
]
