"""Auto utilities.

These are for the various interactive use convenience capabilities of the
trulens-auto package.
"""

import importlib
import logging
from pprint import PrettyPrinter
import subprocess
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

from trulens.core.utils import imports as import_utils
from trulens.core.utils import python as python_utils

# import caller_frame
# from trulens.core.utils.python import caller_module

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

NO_INSTALL: bool = False
"""If set, will not automatically install any optional trulens modules and fail
in the typical way if a module is missing."""


def pip(*args) -> subprocess.CompletedProcess:
    """Execute pip with the given arguments."""

    logger.debug("Running pip", *args)

    return subprocess.run(
        [sys.executable, "-m", "pip", *args], capture_output=True, check=True
    )


def install(package_name: str):
    """Install a package with pip."""

    package_name = import_utils.safe_importlib_package_name(package_name)

    logger.info("Installing package %s", package_name)

    pip("install", package_name)

    return import_utils.get_package_version(package_name)


def make_help_str(
    kinds: Dict[str, Dict],
) -> Tuple[Callable[[], None], Callable[[], str]]:
    """Create a help string that lists all available classes and their installation status."""

    mod = sys.modules[python_utils.caller_frame(offset=1).f_locals["__name__"]]

    def help_str() -> str:
        ret = f"Module '{mod.__name__}' from '{mod.__file__} contains:\n"
        for kind, items in kinds.items():
            ret += f"{kind}: \n"  # TODO: pluralize
            for class_, temp in items.items():
                if len(temp) == 2:
                    package_name, _ = temp
                else:
                    package_name, _, _ = temp
                version = import_utils.get_package_version(package_name)
                ret += (
                    f"  {class_}\t[{package_name}]\t["
                    + (str(version) if version is not None else "not installed")
                    + "]\n"
                )

        if NO_INSTALL:
            ret += "\nYou can enable automatic installs by calling `trulens.auto.set_no_install(False)`."
        else:
            ret += "\nImporting from this module will install the required package. You can disable this by calling `trulens.auto.set_no_install()`."

        return ret

    def help() -> None:
        """Print a help message that lists all available classes and their installation status."""

        print(help_str())

    return help, help_str


def make_getattr_override(
    kinds: Dict[str, Dict],
    help_str: Union[str, Callable[[], str]],
) -> Any:
    """Make a custom __getattr__ function for a module to allow automatic
    installs of missing modules and better error messages."""

    mod = sys.modules[python_utils.caller_module_name(offset=1)]

    if not isinstance(help_str, str):
        help_str: str = help_str()

    if mod.__doc__ is None:
        mod.__doc__ = help_str
    else:
        mod.__doc__ += "\n\n" + help_str

    def getattr_(attr):
        nonlocal mod
        nonlocal help_str

        if attr == "_ipython_display_":
            return lambda: print(help_str)

        if attr in [
            "_ipython_canary_method_should_not_exist_",
            "_ipython_display_",
            "__warningregistry__",
        ]:
            raise AttributeError()

        for kind, options in kinds.items():
            if attr in options:
                if len(options[attr]) == 2:
                    package_name, module_name = options[attr]
                    attr_name = attr
                else:
                    package_name, module_name, attr_name = options[attr]

                try:
                    mod = import_module(
                        package_name=package_name,
                        module_name=module_name,
                        purpose=f"using the {attr} {kind}",
                    )
                    return getattr(mod, attr_name)

                except ImportError as e:
                    raise ImportError(
                        f"""Could not import the {attr} {kind}. You might need to re-install {package_name}:
            ```bash
            pip uninstall -y {package_name}
            pip install {package_name}
            ```
        """
                    ) from e

        # We also need to the case of an import of a sub-module instead of
        # something in __all__ (enumerated in kinds) so we try that here.
        try:
            submod = importlib.import_module(mod.__name__ + "." + attr)
            return submod
        except Exception:
            # Use the same error message as the below.
            pass

        raise AttributeError(
            f"Module {mod.__name__} has no attribute {attr}.\n" + help_str
        )

    return getattr_


def import_module(
    package_name: str, module_name: str, purpose: Optional[str] = None
):
    """Import a module of the given package but install it if it is not already
    installed.
    If purpose is given, will use that for the message printed before the
    installation if it does happen.
    """

    package_name = import_utils.safe_importlib_package_name(package_name)

    if purpose is None:
        purpose = f"using {module_name}"

    try:
        if (package := import_utils.get_package_version(package_name)) is None:
            if NO_INSTALL:
                # Throwing RuntimeError as I don't want this to be caught by the
                # importing try blocks. Needs this to be the final error.
                raise RuntimeError(
                    f"Package {package_name} is not installed. Enable automatic installation by calling `trulens.auto.set_no_install(False)` or install it manually with pip: \n\t```bash\n\tpip install '{package_name}'\n\t```"
                )

            if sys.version_info >= (3, 12) and package_name in [
                "trulens-providers-cortex",
                "trulens-instrument-nemo",
            ]:
                # These require python < 3.12 .
                # RuntimeError here on purpose. See above.
                raise RuntimeError(
                    f"Package {package_name} required for {purpose} does not support python >= 3.12."
                )

            else:
                print(
                    f"Installing package {package_name} required for {purpose} ..."
                )

            package = install(package_name)

    except subprocess.CalledProcessError as e:
        # ImportError here on purpose. Want this to be caught in importing try
        # blocks to give additional information.
        raise ImportError(f"Could not install {package_name}.") from e

    if package is None:
        # Same
        raise ImportError(f"Could not install {package_name}.")

    return importlib.import_module(module_name)
