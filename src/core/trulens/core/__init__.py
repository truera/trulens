"""
# Trulens-eval LLM Evaluation Library

This top-level import includes everything to get started.
"""

import importlib.metadata

from trulens.core.app import TruBasicApp
from trulens.core.app import TruCustomApp
from trulens.core.app import TruVirtual
from trulens.core.feedback.base_feedback import Feedback
from trulens.core.feedback.base_provider import Provider
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.select import Select
from trulens.core.tru import Tru
from trulens.utils.imports import check_imports

check_imports()

__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    "Tru",  # main interface
    # app types
    "TruBasicApp",
    "TruCustomApp",
    "TruVirtual",
    # app setup
    "FeedbackMode",
    # feedback setup
    "Feedback",
    "Select",
    "Provider",
]
