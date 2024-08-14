# ruff: noqa: E402, F822

from typing import TYPE_CHECKING

from trulens.core.utils import imports as import_utils

if TYPE_CHECKING:
    # Needed for static tools:

    # feedback type aliases
    from trulens.core.feedback.feedback import AggCallable

    # feedback classes
    from trulens.core.feedback.feedback import Feedback
    from trulens.core.feedback.feedback import ImpCallable
    from trulens.core.feedback.feedback import InvalidSelector
    from trulens.core.feedback.feedback import SkipEval

    # selector utilities
    from trulens.core.schema import Select
    from trulens.core.schema.feedback import FeedbackCall
    from trulens.core.schema.feedback import FeedbackCombinations
    from trulens.core.schema.feedback import FeedbackDefinition

    # schema enums
    from trulens.core.schema.feedback import FeedbackMode
    from trulens.core.schema.feedback import FeedbackOnMissingParameters

    # schema classes
    from trulens.core.schema.feedback import FeedbackResult
    from trulens.core.schema.feedback import FeedbackResultStatus
    from trulens.feedback.embeddings import Embeddings
    from trulens.feedback.groundtruth import GroundTruthAggregator
    from trulens.feedback.groundtruth import GroundTruthAgreement
    from trulens.feedback.llm_provider import LLMProvider


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
