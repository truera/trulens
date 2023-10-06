import logging
from typing import Any, Callable, Dict, Iterable, Tuple, Union

logger = logging.getLogger(__name__)

# Signature of feedback implementations. Take in any number of arguments
# and return either a single float or a float and a dictionary (of metadata).
ImpCallable = Callable[..., Union[float, Tuple[float, Dict[str, Any]]]]

# Signature of aggregation functions.
AggCallable = Callable[[Iterable[float]], float]

# Main class holding and running feedback functions:
from trulens_eval.feedback.feedback import Feedback
# Specific feedback functions:
from trulens_eval.feedback.embeddings import Embeddings
from trulens_eval.feedback.groundedness import Groundedness
from trulens_eval.feedback.groundtruth import GroundTruthAgreement
from trulens_eval.feedback.provider.cohere import Cohere
# Providers of feedback functions evaluation:
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.feedback.provider.litellm import LiteLLM
from trulens_eval.feedback.provider.openai import AzureOpenAI
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.feedback.provider.bedrock import Bedrock

__all__ = [
    'Feedback', 'Embeddings', 'Groundedness', 'GroundTruthAgreement', 'OpenAI', 'AzureOpenAI',
    'Huggingface', 'Cohere', 'LiteLLM','Bedrock'
]
