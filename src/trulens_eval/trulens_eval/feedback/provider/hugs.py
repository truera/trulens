# ruff: noqa: E401, E402, F401, F403
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.providers.huggingface.provider` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.providers.huggingface.provider import (
    HUGS_CONTEXT_RELEVANCE_API_URL,
)
from trulens.providers.huggingface.provider import HUGS_DOCNLI_API_URL
from trulens.providers.huggingface.provider import HUGS_DOCNLI_MODEL_PATH
from trulens.providers.huggingface.provider import HUGS_HALLUCINATION_API_URL
from trulens.providers.huggingface.provider import HUGS_LANGUAGE_API_URL
from trulens.providers.huggingface.provider import HUGS_NLI_API_URL
from trulens.providers.huggingface.provider import HUGS_PII_DETECTION_API_URL
from trulens.providers.huggingface.provider import HUGS_SENTIMENT_API_URL
from trulens.providers.huggingface.provider import HUGS_TOXIC_API_URL
from trulens.providers.huggingface.provider import Dummy
from trulens.providers.huggingface.provider import Huggingface
from trulens.providers.huggingface.provider import HuggingfaceBase
from trulens.providers.huggingface.provider import HuggingfaceLocal
