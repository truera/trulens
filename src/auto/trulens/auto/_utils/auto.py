"""Auto utilities.

These are for the various interactive use convenience capabilities of the
trulens-auto package.
"""

import importlib
import logging
from pprint import PrettyPrinter
import subprocess
import sys
from types import ModuleType
from typing import Any, Dict, Optional

from IPython.lib import pretty
from trulens.core.utils import imports as import_utils
from trulens.core.utils import python as python_utils

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

NO_INSTALL: bool = False
"""If set, will not automatically install any optional trulens modules and fail
in the typical way if a module is missing."""

DOC_NO_INSTALLS: str = "You can enable automatic installs by calling `trulens.auto.set_no_install(False)`."
"""Instructions for enabling automatic installs."""

DOC_INSTALLS = (
    "Importing from this module will install the required package. "
    "You can disable this by calling `trulens.auto.set_no_install()`."
)
"""Instructions for disabling automatic installs."""


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


def make_help_ipython(
    doc: Optional[str] = None,
    kinds: Optional[Dict[str, Dict]] = None,
    kinds_docs: Optional[Dict[str, str]] = None,
    mod: Optional[ModuleType] = None,
) -> pretty.Printable:
    """Create a help string that lists all available classes and their installation status."""

    if mod is None:
        mod: ModuleType = sys.modules[
            python_utils.caller_frame(offset=1).f_locals["__name__"]
        ]

    if kinds_docs is None:
        kinds_docs = {}

    class _PrettyModule(pretty.Printable):
        def help(self) -> None:
            print(self.__doc__)

        def __repr__(self) -> str:
            nonlocal doc, mod, kinds, kinds_docs

            ret = f"Module '{mod.__name__}' from '{mod.__file__}\n"
            if doc is not None:
                ret += doc + "\n\n"
            else:
                ret += "\n"

            if kinds is None:
                return ret

            for kind, items in kinds.items():
                ret += f"{kind.capitalize()}: \n"
                if kind in kinds_docs:
                    ret += f"{kinds_docs[kind]}\n\n"

                for class_, temp in items.items():
                    if len(temp) == 2:
                        package_name, _ = temp
                    else:
                        package_name, _, _ = temp
                    version = import_utils.get_package_version(package_name)
                    ret += (
                        f"  {class_}\t[{package_name}]\t["
                        + (
                            str(version)
                            if version is not None
                            else "not installed"
                        )
                        + "]\n"
                    )

                ret += "\n"

            if NO_INSTALL:
                ret += DOC_NO_INSTALLS
            else:
                ret += DOC_INSTALLS

            return ret

        def _repr_markdown_(self) -> str:
            nonlocal doc, mod, kinds, kinds_docs

            ret = f"# Module `{mod.__name__}`\n"
            ret += f"from `{mod.__file__}`\n\n"

            if doc is not None:
                ret += f"{doc}\n\n"

            if kinds is None:
                return ret

            for kind, items in kinds.items():
                ret += f"## {kind.capitalize()}\n"
                if kind in kinds_docs:
                    ret += f"{kinds_docs[kind]}\n\n"

                ret += f"{kind} | package | installed version\n"
                ret += "--- | --- | ---\n"
                for class_, temp in items.items():
                    if len(temp) == 2:
                        package_name, _ = temp
                    else:
                        package_name, _, _ = temp
                    version = import_utils.get_package_version(package_name)
                    ret += (
                        f"`{class_}` | `{package_name}` | "
                        + (
                            str(version)
                            if version is not None
                            else "not installed"
                        )
                        + "\n"
                    )
                ret += "\n"

            if NO_INSTALL:
                ret += "\n" + DOC_INSTALLS
            else:
                ret += "\n" + DOC_NO_INSTALLS

            return ret

        def _repr_pretty_(self, p: pretty.PrettyPrinter, cycle: bool) -> None:
            nonlocal doc, mod, kinds, kinds_docs

            p.text(f"{mod.__name__}")
            p.break_()

            if doc is not None:
                p.text(doc)
                p.break_()
                p.break_()

            if kinds is None:
                return

            p.text("Contents:")
            p.break_()
            p.break_()

            for kind, items in kinds.items():
                p.begin_group(indent=2, open=kind.capitalize())
                p.break_()
                if kind in kinds_docs:
                    p.text(kinds_docs[kind])
                    p.break_()
                    p.break_()

                p.text(f"{kind} | package | installed version")
                p.break_()
                p.text("--- | --- | ---")
                p.break_()
                for class_, temp in items.items():
                    if len(temp) == 2:
                        package_name, _ = temp
                    else:
                        package_name, _, _ = temp
                    version = import_utils.get_package_version(package_name)
                    p.text(
                        f"`{class_}` | `{package_name}` | "
                        + (
                            str(version)
                            if version is not None
                            else "not installed"
                        )
                    )
                    p.break_()

                p.break_()
                p.end_group(dedent=2)

            if NO_INSTALL:
                p.text(DOC_NO_INSTALLS)
                p.break_()
            else:
                p.text(DOC_INSTALLS)
                p.break_()

    p = _PrettyModule()
    p.__doc__ = repr(p)
    p.__name__ = mod.__name__

    return p


def make_getattr_override(
    doc: Optional[str] = None,
    kinds: Optional[Dict[str, Dict]] = None,
    kinds_docs: Optional[Dict[str, str]] = None,
    mod: Optional[ModuleType] = None,
) -> Any:
    """Make a custom __getattr__ function for a module to allow automatic
    installs of missing modules and better error messages."""

    if mod is None:
        mod: ModuleType = sys.modules[
            python_utils.caller_frame(offset=1).f_locals["__name__"]
        ]

    if kinds_docs is None:
        kinds_docs = {}

    pretties = make_help_ipython(
        doc=doc, kinds=kinds, kinds_docs=kinds_docs, mod=mod
    )

    if mod.__doc__ is None:
        mod.__doc__ = pretties.__doc__
    else:
        mod.__doc__ += "\n\n" + pretties.__doc__

    def getattr_(attr):
        nonlocal mod, pretties

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
        # try:
        #    submod = importlib.import_module(mod.__name__ + "." + attr)
        #    return submod
        # except Exception:
        # Use the same error message as the below.
        #    pass

        if hasattr(pretties, attr):
            # Use the represtations from the PrettyModule class.
            return getattr(pretties, attr)

        raise AttributeError(
            f"Module {mod.__name__} has no attribute {attr}.\n" + repr(pretties)
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
