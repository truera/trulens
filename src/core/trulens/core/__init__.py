"""Trulens Core LLM Evaluation Library.

The `trulens-core` library includes everything to get started.
"""

from importlib.metadata import version
import os

from trulens.core import session as mod_session
from trulens.core.feedback import feedback as mod_feedback
from trulens.core.feedback import provider as mod_provider
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import select as select_schema
from trulens.core.utils import imports as import_utils

# NOTE: workaround for MKL and multiprocessing
# https://github.com/pytorch/csprng/issues/115
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)


TruSession = mod_session.TruSession
FeedbackMode = feedback_schema.FeedbackMode
Feedback = mod_feedback.Feedback
SnowflakeFeedback = mod_feedback.SnowflakeFeedback
Select = select_schema.Select
Provider = mod_provider.Provider
Tru = TruSession


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
