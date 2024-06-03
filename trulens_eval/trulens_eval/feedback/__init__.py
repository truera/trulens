"""Aliases for importing things related to feedback functions."""

# Specific feedback functions:
# Main class holding and running feedback functions:
from trulens_eval.feedback import feedback as mod_feedback
from trulens_eval.feedback import embeddings as mod_embeddings
from trulens_eval.feedback import groundtruth as mod_groundtruth
# Providers of feedback functions evaluation:
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.feedback.provider.langchain import Langchain
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_BEDROCK
from trulens_eval.utils.imports import REQUIREMENT_LITELLM
from trulens_eval.utils.imports import REQUIREMENT_OPENAI

with OptionalImports(messages=REQUIREMENT_BEDROCK):
    from trulens_eval.feedback.provider.bedrock import Bedrock

with OptionalImports(messages=REQUIREMENT_LITELLM):
    from trulens_eval.feedback.provider.litellm import LiteLLM

with OptionalImports(messages=REQUIREMENT_OPENAI):
    from trulens_eval.feedback.provider.openai import AzureOpenAI
    from trulens_eval.feedback.provider.openai import OpenAI

Feedback = mod_feedback.Feedback
Embeddings = mod_embeddings.Embeddings
GroundTruthAgreement = mod_groundtruth.GroundTruthAgreement

__all__ = [
    "Feedback",
    "Embeddings",
    "GroundTruthAgreement",
    "OpenAI",
    "AzureOpenAI",
    "Huggingface",
    "LiteLLM",
    "Bedrock",
    "Langchain",
]
