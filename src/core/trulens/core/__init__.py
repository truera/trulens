# TODO: No aliases in production API.

"""Trulens Core LLM Evaluation Library

The `trulens-core` library includes everything to get started.
"""

import os

# NOTE: workaround for MKL and multiprocessing
# https://github.com/pytorch/csprng/issues/115
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from importlib.metadata import version

from trulens.core.app.basic import TruBasicApp
from trulens.core.app.custom import TruCustomApp
from trulens.core.app.virtual import TruVirtual
from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.feedback import SnowflakeFeedback
from trulens.core.feedback.provider import Provider
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.select import Select
from trulens.core.tru import Tru
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    # main interface
    "Tru",
    # app types
    "TruBasicApp",
    "TruCustomApp",
    "TruVirtual",
    # app setup
    "FeedbackMode",
    # feedback setup
    "Feedback",
    "SnowflakeFeedback",
    "Select",
    "Provider",
]
