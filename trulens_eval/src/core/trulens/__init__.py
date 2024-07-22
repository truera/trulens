"""
# Trulens-eval LLM Evaluation Library

This top-level import includes everything to get started.
"""

import importlib.metadata

from trulens import tru as mod_tru
from trulens import tru_basic_app as mod_tru_basic_app
from trulens import tru_custom_app as mod_tru_custom_app
from trulens import tru_virtual as mod_tru_virtual
from trulens.utils import threading as mod_threading_utils

# This check is intentionally done ahead of the other imports as we want to
# print out a nice warning/error before an import error happens further down
# this sequence.
# from trulens.utils.imports import check_imports

# check_imports()

# from trulens import tru_chain as mod_tru_chain
# from trulens.feedback import feedback as mod_feedback
# from trulens.feedback.provider import base as mod_provider
# from trulens.feedback.provider import hugs as mod_hugs_provider
# from trulens.feedback.provider import langchain as mod_langchain_provider
# from trulens.schema import feedback as mod_feedback_schema
# from trulens.utils import imports as mod_imports_utils

__version__ = importlib.metadata.version(__package__ or __name__)

# Optional provider types.

# with mod_imports_utils.OptionalImports(
#         messages=mod_imports_utils.REQUIREMENT_LITELLM):
#     from trulens.feedback.provider.litellm import LiteLLM

# with mod_imports_utils.OptionalImports(
#         messages=mod_imports_utils.REQUIREMENT_BEDROCK):
#     from trulens.feedback.provider.bedrock import Bedrock

# with mod_imports_utils.OptionalImports(
#         messages=mod_imports_utils.REQUIREMENT_OPENAI):
#     from trulens.feedback.provider.openai import AzureOpenAI
#     from trulens.feedback.provider.openai import OpenAI

# # Optional app types.

# with mod_imports_utils.OptionalImports(
#         messages=mod_imports_utils.REQUIREMENT_LLAMA):
#     from trulens.tru_llama import TruLlama

# with mod_imports_utils.OptionalImports(
#         messages=mod_imports_utils.REQUIREMENT_RAILS):
#     from trulens.tru_rails import TruRails

# # the dependency snowflake-snowpark-python not yet supported in 3.12
# with mod_imports_utils.OptionalImports(
#         messages=mod_imports_utils.REQUIREMENT_CORTEX):
#     from trulens.feedback.provider.cortex import Cortex

Tru = mod_tru.Tru
TruBasicApp = mod_tru_basic_app.TruBasicApp
# TruChain = mod_tru_chain.TruChain
TruCustomApp = mod_tru_custom_app.TruCustomApp
TruVirtual = mod_tru_virtual.TruVirtual
TP = mod_threading_utils.TP
# Feedback = mod_feedback.Feedback
# Provider = mod_provider.Provider
# Huggingface = mod_hugs_provider.Huggingface
# HuggingfaceLocal = mod_hugs_provider.HuggingfaceLocal
# Langchain = mod_langchain_provider.Langchain
# FeedbackMode = mod_feedback_schema.FeedbackMode
# Select = mod_feedback_schema.Select

__all__ = [
    'Tru',  # main interface

    # app types
    'TruBasicApp',
    'TruCustomApp',
    'TruVirtual',
    # "TruChain",
    # "TruLlama",
    # "TruRails",

    # app setup
    'FeedbackMode',

    # feedback setup
    # "Feedback",
    # "Select",

    # feedback providers
    # "Provider",
    # "AzureOpenAI",
    # "OpenAI",
    # "Langchain",
    # "LiteLLM",
    # "Bedrock",
    # "Huggingface",
    # "HuggingfaceLocal",
    # "Cortex",

    # misc utility
    'TP',
]
