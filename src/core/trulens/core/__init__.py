"""TruLens Core LLM Evaluation Library."""

# WARNING: This file does not follow the no-init aliases import standard.

from importlib.metadata import version
import os

from trulens.core.feedback.feedback import Feedback
from trulens.core.feedback.feedback import SnowflakeFeedback
from trulens.core.feedback.provider import Provider
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.select import Select
from trulens.core.session import Tru
from trulens.core.session import TruSession
from trulens.core.utils import imports as import_utils

# NOTE: workaround for MKL and multiprocessing
# https://github.com/pytorch/csprng/issues/115
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)

__all__ = [
    "Tru",  # aliases TruSession for backwards compatibility
    "TruSession",  # main interface
    # app setup
    "FeedbackMode",
    # feedback setup
    "Feedback",
    "SnowflakeFeedback",
    "Select",
    "Provider",
]
