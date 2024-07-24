"""
# Trulens-eval LLM Evaluation Library

This top-level import includes everything to get started.
"""

import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from trulens.core.feedback.base_feedback import Feedback
from trulens.core.feedback.base_provider import Provider
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.feedback import Select
from trulens.core.tru import Tru
from trulens.core.tru_basic_app import TruBasicApp
from trulens.core.tru_custom_app import TruCustomApp
from trulens.core.tru_virtual import TruVirtual
from trulens.utils.imports import check_imports

check_imports()


__all__ = [
    'Tru',  # main interface

    # app types
    'TruBasicApp',
    'TruCustomApp',
    'TruVirtual',

    # app setup
    'FeedbackMode',

    # feedback setup
    'Feedback',
    'Select',
    'Provider'
]
