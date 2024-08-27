# ruff: noqa: E402, F822
"""TruLens feedback function specification and implementation."""

from typing import TYPE_CHECKING

from trulens.auto._utils import auto as auto_utils

if TYPE_CHECKING:
    # Needed for static tools:

    from trulens.core.feedback.feedback import AggCallable
    from trulens.core.feedback.feedback import Feedback
    from trulens.core.feedback.feedback import ImpCallable
    from trulens.core.feedback.feedback import SkipEval
    from trulens.core.feedback.provider import Provider
    from trulens.core.schema import Select
    from trulens.core.schema.feedback import FeedbackCall
    from trulens.core.schema.feedback import FeedbackCombinations
    from trulens.core.schema.feedback import FeedbackDefinition
    from trulens.core.schema.feedback import FeedbackMode
    from trulens.core.schema.feedback import FeedbackOnMissingParameters
    from trulens.core.schema.feedback import FeedbackResult
    from trulens.core.schema.feedback import FeedbackResultStatus
    from trulens.feedback.llm_provider import LLMProvider

_SPECS = {
    "Feedback": ("trulens-core", "trulens.feedback.feedback"),
    "Select": ("trulens-core", "trulens.core.schema"),
    "FeedbackOnMissingParameters": (
        "trulens-core",
        "trulens.core.schema.feedback",
    ),
    "FeedbackCombinations": ("trulens-core", "trulens.core.schema.feedback"),
}

_CONFIGS = {
    "FeedbackMode": ("trulens-core", "trulens.core.schema.feedback"),
}

_IMPS = {
    "SkipEval": ("trulens-core", "trulens.feedback.feedback"),
    "ImpCallable": ("trulens-core", "trulens.feedback.feedback"),
    "AggCallable": ("trulens-core", "trulens.feedback.feedback"),
}

_RESULTS = {
    "FeedbackResultStatus": ("trulens-core", "trulens.core.schema.feedback"),
    "FeedbackResult": ("trulens-core", "trulens.core.schema.feedback"),
    "FeedbackCall": ("trulens-core", "trulens.core.schema.feedback"),
}

_INTERFACES = {
    "Provider": ("trulens-core", "trulens.core.feedback.provider"),
    "LLMProvider": ("trulens-core", "trulens.feedback.llm_provider"),
    "FeedbackDefinition": ("trulens-core", "trulens.core.schema.feedback"),
}

_KINDS = {
    "spec": _SPECS,
    "config": _CONFIGS,
    "implementation": _IMPS,
    "result": _RESULTS,
    "interface": _INTERFACES,
}

__getattr__ = auto_utils.make_getattr_override(kinds=_KINDS)

__all__ = [
    # feedback specification
    "Feedback",
    "Select",
    "FeedbackOnMissingParameters",
    "FeedbackCombinations",
    # feedback configuration
    "FeedbackMode",
    # feedback implementation
    "SkipEval",
    "ImpCallable",
    "AggCallable",
    # feedback result enums/schemas
    "FeedbackResult",
    "FeedbackResultStatus",
    "FeedbackCall",
    # interfaces
    "FeedbackDefinition",
    "Provider",
    "LLMProvider",
]
