"""
# Trulens-eval LLM Evaluation Library

This top-level import includes everything to get started.
"""

import importlib.metadata

from trulens.feedback.base_feedback import Feedback
from trulens.feedback.base_provider import Provider
from trulens.schema.feedback import FeedbackMode
from trulens.schema.feedback import Select
from trulens.tru import Tru
from trulens.tru_basic_app import TruBasicApp
from trulens.tru_custom_app import TruCustomApp
from trulens.tru_virtual import TruVirtual
# This check is intentionally done ahead of the other imports as we want to
# print out a nice warning/error before an import error happens further down
# this sequence.
from trulens.utils.imports import check_imports
from trulens.utils.threading import TP

__version__ = importlib.metadata.version(__package__ or __name__)

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

    # misc utility
    'TP',
]
