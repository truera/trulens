"""Trulens Core LLM Evaluation Library."""

from importlib.metadata import version
import os

from trulens.core.utils import imports as import_utils

# NOTE: workaround for MKL and multiprocessing
# https://github.com/pytorch/csprng/issues/115
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)
