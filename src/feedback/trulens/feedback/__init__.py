# Specific feedback functions:
# Main class holding and running feedback functions:
from importlib.metadata import version

from trulens.core.utils import imports as import_utils

__version__ = version(
    import_utils.safe_importlib_package_name(__package__ or __name__)
)
#
