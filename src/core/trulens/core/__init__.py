"""# Trulens Core LLM Evaluation Library

The `trulens-core` library includes everything to get started.

"""

import os

# NOTE: workaround for MKL and multiprocessing
# https://github.com/pytorch/csprng/issues/115
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from importlib.metadata import version

from trulens.core.app import TruBasicApp
from trulens.core.app import TruCustomApp
from trulens.core.app import TruVirtual
from trulens.core.feedback import Feedback
from trulens.core.feedback import Provider
from trulens.core.feedback import SnowflakeFeedback
from trulens.core.schema import FeedbackMode
from trulens.core.schema import Select
from trulens.core.tru import TruSession
from trulens.core.utils.imports import safe_importlib_package_name

__version__ = version(safe_importlib_package_name(__package__ or __name__))


__all__ = [
    "TruSession",  # main interface
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
