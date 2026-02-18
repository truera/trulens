"""TruLens Core LLM Evaluation Library."""

# WARNING: This file does not follow the no-init aliases import standard.

# Suppress third-party library warnings before importing anything else
import logging
import warnings

# Suppress pkg_resources deprecation warning from munch and other libraries
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

# Suppress python-dotenv parsing warnings for malformed .env files
warnings.filterwarnings(
    "ignore",
    message="python-dotenv could not parse statement",
    category=UserWarning,
)

# Suppress python-dotenv logger warnings for malformed .env files
logging.getLogger("dotenv.main").setLevel(logging.ERROR)

from importlib.metadata import version  # noqa: E402
import os  # noqa: E402

from trulens.core.feedback.feedback import Feedback  # noqa: E402
from trulens.core.feedback.feedback import SnowflakeFeedback  # noqa: E402
from trulens.core.feedback.provider import Provider  # noqa: E402
from trulens.core.metric import Metric  # noqa: E402
from trulens.core.metric import Selector  # noqa: E402
from trulens.core.schema.feedback import FeedbackMode  # noqa: E402
from trulens.core.schema.select import Select  # noqa: E402
from trulens.core.session import Tru  # noqa: E402
from trulens.core.session import TruSession  # noqa: E402
from trulens.core.utils import imports as import_utils  # noqa: E402

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
    # metric setup (new unified API)
    "Metric",
    "Selector",
    # feedback setup (deprecated, use Metric instead)
    "Feedback",
    "SnowflakeFeedback",
    "Select",
    "Provider",
]
