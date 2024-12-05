from importlib.metadata import version
import sys


def safe_importlib_package_name(package_name: str) -> str:
    """Convert a package name that may have periods in it to one that uses
    hyphens for periods but only if the python version is old.
    Copied from trulens-core to avoid a circular dependency."""

    return (
        package_name
        if sys.version_info >= (3, 10)
        else package_name.replace(".", "-")
    )


__version__ = version(safe_importlib_package_name(__package__ or __name__))
