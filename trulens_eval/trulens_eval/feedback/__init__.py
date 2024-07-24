"""Aliases for importing things related to feedback functions."""

# Specific feedback functions:
# Main class holding and running feedback functions:
from trulens_eval.feedback import embeddings as mod_embeddings
from trulens_eval.feedback import feedback as mod_feedback
from trulens_eval.feedback import groundtruth as mod_groundtruth

# Providers of feedback functions evaluation:
from trulens_eval.feedback.provider import hugs as mod_hugs_provider
from trulens_eval.feedback.provider import langchain as mod_langchain_provider
from trulens_eval.utils import imports as import_utils

with import_utils.OptionalImports(messages=import_utils.REQUIREMENT_BEDROCK):
    from trulens_eval.feedback.provider.bedrock import Bedrock

with import_utils.OptionalImports(messages=import_utils.REQUIREMENT_LITELLM):
    from trulens_eval.feedback.provider.litellm import LiteLLM

with import_utils.OptionalImports(messages=import_utils.REQUIREMENT_OPENAI):
    from trulens_eval.feedback.provider.openai import AzureOpenAI
    from trulens_eval.feedback.provider.openai import OpenAI

with import_utils.OptionalImports(messages=import_utils.REQUIREMENT_CORTEX):
    from trulens_eval.feedback.provider.cortex import Cortex

Feedback = mod_feedback.Feedback
Embeddings = mod_embeddings.Embeddings
GroundTruthAgreement = mod_groundtruth.GroundTruthAgreement

Huggingface = mod_hugs_provider.Huggingface
HuggingfaceLocal = mod_hugs_provider.HuggingfaceLocal

Langchain = mod_langchain_provider.Langchain

__all__ = [
    "Feedback",
    "Embeddings",
    "GroundTruthAgreement",
    "OpenAI",
    "AzureOpenAI",
    "Huggingface",
    "HuggingfaceLocal",
    "LiteLLM",
    "Bedrock",
    "Langchain",
    "Cortex",
]
