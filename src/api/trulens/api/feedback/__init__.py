# ruff: noqa: E402, F822

from typing import TYPE_CHECKING

from trulens.core.utils import imports as import_utils

if TYPE_CHECKING:
    from trulens.feedback import embeddings as mod_embeddings
    from trulens.feedback import groundtruth as mod_groundtruth
    from trulens.feedback import llm_provider as mod_llm_provider

    Embeddings = mod_embeddings.Embeddings
    GroundTruthAggregator = mod_groundtruth.GroundTruthAggregator
    GroundTruthAgreement = mod_groundtruth.GroundTruthAgreement
    LLMProvider = mod_llm_provider.LLMProvider

    # NOTE: This is only needed for static tools to figure out the content of
    # this module because it is dynamically determined during runtime.

    embeddings = mod_embeddings
    groundtruth = mod_groundtruth
    llm_provider = mod_llm_provider

_CLASSES = {
    "Embeddings": ("trulens-feedback", "trulens.feedback.embeddings"),
    "GroundTruthAggregator": (
        "trulens-feedback",
        "trulens.feedback.groundtruth",
    ),
    "GroundTruthAgreement": (
        "trulens-feedback",
        "trulens.feedback.llm_provider",
    ),
    "LLMProvider": ("trulens-feedback", "trulens.feedback.llm_provider"),
    "Feedback": ("trulens-core", "trulens.feedback.feedback"),
    "InvalidSelector": ("trulens-core", "trulens.feedback.feedback"),
    "SkipEval": ("trulens-core", "trulens.feedback.feedback"),
}

_TYPES = {
    "ImpCallable": ("trulens-core", "trulens.feedback.feedback"),
    "AggCallable": ("trulens-core", "trulens.feedback.feedback"),
}

_ENUMS = {
    "FeedbackMode": ("trulens-core", "trulens.core.schema.feedback"),
    "FeedbackResultStatus": ("trulens-core", "trulens.core.schema.feedback"),
    "FeedbackOnMissingParameters": (
        "trulens-core",
        "trulens.core.schema.feedback",
    ),
    "FeedbackCombinations": ("trulens-core", "trulens.core.schema.feedback"),
}

_SCHEMAS = {
    "FeedbackResult": ("trulens-core", "trulens.core.schema.feedback"),
    "FeedbackCall": ("trulens-core", "trulens.core.schema.feedback"),
    "FeedbackDefinition": ("trulens-core", "trulens.core.schema.feedback"),
}

_UTILITIES = {
    "Select": ("trulens-core", "trulens.core.schema"),
}

_KINDS = {
    "class": _CLASSES,
    "type": _TYPES,
    "enum": _ENUMS,
    "schema": _SCHEMAS,
    "utility": _UTILITIES,
}

if TYPE_CHECKING:
    from trulens.core import schema as core_schema
    from trulens.core.feedback import feedback as mod_feedback
    from trulens.core.schema import feedback as feedback_schema

    # feedback classes
    Feedback = mod_feedback.Feedback
    InvalidSelector = mod_feedback.InvalidSelector
    SkipEval = mod_feedback.SkipEval

    # feedback type aliases
    ImpCallable = mod_feedback.ImpCallable
    AggCallable = mod_feedback.AggCallable

    # schema enums
    FeedbackMode = feedback_schema.FeedbackMode
    FeedbackResultStatus = feedback_schema.FeedbackResultStatus
    FeedbackOnMissingParameters = feedback_schema.FeedbackOnMissingParameters
    FeedbackCombinations = feedback_schema.FeedbackCombinations

    # schema classes
    FeedbackResult = feedback_schema.FeedbackResult
    FeedbackCall = feedback_schema.FeedbackCall
    FeedbackDefinition = feedback_schema.FeedbackDefinition

    # selector utilities
    Select = core_schema.Select

help, help_str = import_utils.make_help_str(_KINDS)

__getattr__ = import_utils.make_getattr_override(_KINDS, help_str=help_str)

__all__ = [
    "Embeddings",
    "GroundTruthAgreement",
    "GroundTruthAggregator",
    "LLMProvider",
    "Feedback",
    "FeedbackMode",
    "FeedbackResult",
    "FeedbackResultStatus",
    "FeedbackCall",
    "FeedbackDefinition",
    "FeedbackOnMissingParameters",
    "FeedbackCombinations",
    "InvalidSelector",
    "SkipEval",
    "Select",
    "ImpCallable",
    "AggCallable",
]
