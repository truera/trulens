import logging
from typing import Any, Callable, Dict, Iterable, Tuple, Union

from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_BEDROCK
from trulens_eval.utils.imports import REQUIREMENT_LITELLM
from trulens_eval.utils.imports import REQUIREMENT_OPENAI

logger = logging.getLogger(__name__)

# Signature of feedback implementations. Take in any number of arguments
# and return either a single float or a float and a dictionary (of metadata).
ImpCallable = Callable[..., Union[float, Tuple[float, Dict[str, Any]]]]

# Signature of aggregation functions.
AggCallable = Callable[[Iterable[float]], float]

# Specific feedback functions:
from trulens_eval.feedback.embeddings import Embeddings
# Main class holding and running feedback functions:
from trulens_eval.feedback.feedback import Feedback
from trulens_eval.feedback.groundedness import Groundedness
from trulens_eval.feedback.groundtruth import GroundTruthAgreement
# Providers of feedback functions evaluation:
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.feedback.provider.langchain import Langchain

with OptionalImports(messages=REQUIREMENT_BEDROCK):
    from trulens_eval.feedback.provider.bedrock import Bedrock

with OptionalImports(messages=REQUIREMENT_LITELLM):
    from trulens_eval.feedback.provider.litellm import LiteLLM

with OptionalImports(messages=REQUIREMENT_OPENAI):
    from trulens_eval.feedback.provider.openai import AzureOpenAI
    from trulens_eval.feedback.provider.openai import OpenAI

__all__ = [
    "Feedback",
    "Embeddings",
    "Groundedness",
    "GroundTruthAgreement",
    "OpenAI",
    "AzureOpenAI",
    "Huggingface",
    "Cohere",
    "LiteLLM",
    "Bedrock",
    "Langchain",
]
