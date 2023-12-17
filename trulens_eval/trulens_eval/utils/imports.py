"""
Utilities for importing python modules and optional importing.
"""

import builtins
from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
from pprint import PrettyPrinter
from typing import Dict, Optional, Sequence, Type, Union

import pkg_resources

from trulens_eval import __name__ as trulens_name

logger = logging.getLogger(__name__)
pp = PrettyPrinter()


def requirements_of_file(path: Path) -> Dict[str, pkg_resources.Requirement]:
    reqs = pkg_resources.parse_requirements(path.read_text())
    mapping = {req.project_name: req for req in reqs}

    return mapping


required_packages = requirements_of_file(
    Path(pkg_resources.resource_filename("trulens_eval", "requirements.txt"))
)
optional_packages = requirements_of_file(
    Path(
        pkg_resources.resource_filename(
            "trulens_eval", "requirements.optional.txt"
        )
    )
)

all_packages = {**required_packages, **optional_packages}

def pin_spec(r: pkg_resources.Requirement) -> pkg_resources.Requirement:
    """
    Pin the requirement to the version assuming it is lower bounded by a
    version.
    """
    
    spec = str(r)
    if ">=" not in spec:
        raise ValueError(f"Requirement {spec} is not lower-bounded.")

    spec = spec.replace(">=", "==")
    return pkg_resources.Requirement.parse(spec)

@dataclass
class ImportErrorMessages():
    module_not_found: str
    import_error: str


def format_import_errors(
    packages: Union[str, Sequence[str]],
    purpose: Optional[str] = None,
    throw: Union[bool, Exception] = False
) -> ImportErrorMessages:
    """
    Format two messages for missing optional package or bad import from an
    optional package.. Throws an `ImportError` with the formatted message if
    `throw` flag is set. If `throw` is already an exception, throws that instead
    after printing the message.
    """

    if purpose is None:
        purpose = f"using {packages}"

    if isinstance(packages, str):
        packages = [packages]

    requirements = []
    requirements_pinned = []

    for pkg in packages:
        if pkg in all_packages:
            requirements.append(str(all_packages[pkg]))
            requirements_pinned.append(str(pin_spec(all_packages[pkg])))
        else:
            print(f"WARNING: package {pkg} not present in requirements.")
            requirements.append(pkg)

    packs = ','.join(packages)
    pack_s = "package" if len(packages) == 1 else "packages"
    is_are = "is" if len(packages) == 1 else "are"
    it_them = "it" if len(packages) == 1 else "them"
    this_these = "this" if len(packages) == 1 else "these"

    msg = (f"""
{','.join(packages)} {pack_s} {is_are} required for {purpose}.
You should be able to install {it_them} with pip:

    pip install '{' '.join(requirements)}
""")

    msg_pinned = (
f"""
You have {packs} installed but we could not import the required
components. There may be a version incompatibility. Please try installing {this_these}
exact {pack_s} with pip: 

  pip install '{' '.join(requirements_pinned)}'

Alternatively, if you do not need {packs}, uninstall {it_them}:

  pip uninstall '{' '.join(packages)}'
"""
    )

    if isinstance(throw, Exception):
        print(msg)
        raise throw

    elif isinstance(throw, bool):
        if throw:
            raise ImportError(msg)

    return ImportErrorMessages(
        module_not_found=msg,
        import_error=msg_pinned
    )

REQUIREMENT_LLAMA = format_import_errors(
    'llama-index', purpose="instrumenting llama_index apps"
)

REQUIREMENT_LANGCHAIN = format_import_errors(
    'langchain', purpose="instrumenting langchain apps"
)

REQUIREMENT_SKLEARN = format_import_errors(
    "scikit-learn", purpose="using embedding vector distances"
)

REQUIREMENT_BEDROCK = format_import_errors(
    ['boto3', 'botocore'], purpose="using Bedrock models"
)

REQUIREMENT_OPENAI = format_import_errors(
    'openai', purpose="using OpenAI models"
)

REQUIREMENT_BERT_SCORE = format_import_errors(
    "bert-score", purpose="measuring BERT Score"
)

REQUIREMENT_EVALUATE = format_import_errors(
    "evaluate", purpose="using certain metrics"
)


class Dummy(object):
    """
    Class to pretend to be a module or some other imported object. Will raise an
    error if accessed in any way.
    """

    def __init__(
        self,
        message: str,
        exception_class: Type[Exception] = ModuleNotFoundError,
        importer=None
    ):
        self.message = message
        self.importer = importer
        self.exception_class = exception_class

    def __call__(self, *args, **kwargs):
        raise self.exception_class(self.message)

    def __getattr__(self, name):
        # If in OptionalImport context, create a new dummy for the requested
        # attribute. Otherwise raise error.

        if self.importer is not None and self.importer.importing:
            return Dummy(message=self.message, importer=self.importer)

        raise self.exception_class(self.message)


class OptionalImports(object):
    """
    Helper context manager for doing multiple imports from an optional module:

    ```python
        messages = ImportErrorMessages(
            module_not_found="install llama_index first",
            import_error="install llama_index==0.1.0"
        )
        with OptionalImports(messages=messages):
            import llama_index
            from llama_index import query_engine
    ```

    The above python block will not raise any errors but once anything else
    about llama_index or query_engine gets accessed, an error is raised with the
    specified message (unless llama_index is installed of course).
    """

    def __init__(self, messages: ImportErrorMessages):
        self.messages = messages
        self.importing = False
        self.imp = builtins.__import__

    def __import__(self, name, globals=None, locals=None, fromlist=(), level=0):
        try:
            mod = self.imp(name, globals, locals, fromlist, level)

            # NOTE(piotrm): commented block attempts to check module contents
            # for required attributes so we can offer a message without raising
            # an exception later. It is commented out for now it is catching
            # some things we don't watch to catch. Need to check how attributes
            # are normally looked up in a module to fix this. Specifically, the
            # code that raises these messages: "ImportError: cannot import name
            # ..."
            """
            if isinstance(fromlist, Iterable):
                for i in fromlist:
                    if i == "*":
                        continue
                    # Check the contents so there is opportunity to catch import errors here
                    try:
                        getattr(mod, i)
                    except AttributeError as e:
                        raise ImportError(e)
            """

            return mod

        except ModuleNotFoundError as e:
            # Check if the import error was from an import in trulens_eval as
            # otherwise we don't want to intercept the error as some modules
            # rely on import failures for various things.
            module_name = inspect.currentframe().f_back.f_globals["__name__"]
            if not module_name.startswith(trulens_name):
                raise e
            logger.debug(f"Module not found {name}.")
            return Dummy(message=self.messages.module_not_found, importer=self)
        
        # NOTE(piotrm): This below seems to never be caught. It might be that a
        # different import method is being used once a module is found.
        """
        except ImportError as e:
            module_name = inspect.currentframe().f_back.f_globals["__name__"]
            if not module_name.startswith(trulens_name):
                raise e
            logger.debug(f"Could not import {name} ({fromlist}). There might be a version incompatibility.")
            logger.warning(self.messages.import_error + "\n" + str(e))

            return Dummy(
                message=self.messages.import_error,
                exception_class=ImportError,
                importer=self
            )
        """

    def __enter__(self):
        builtins.__import__ = self.__import__
        self.importing = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.importing = False
        builtins.__import__ = self.imp

        if exc_value is None:
            return None

        # Print the appropriate message.

        if isinstance(exc_value, ModuleNotFoundError):
            print(exc_value)
            print(self.messages.module_not_found)
        elif isinstance(exc_value, ImportError):
            print(exc_value)
            print(self.messages.import_error)
        else:
            print(exc_value)

        # Will re-raise exception unless True is returned.
        return None
