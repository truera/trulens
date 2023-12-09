"""
Utilities for importing python modules and optional importing.
"""

import builtins
import inspect
import logging
from pathlib import Path
from pprint import PrettyPrinter
from typing import Dict, Iterable, Sequence, Union

import pkg_resources

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

def requirements_of_file(path: Path) -> Dict[str, pkg_resources.Requirement]:
    reqs = pkg_resources.parse_requirements(
        path.read_text()
    )
    mapping = dict()
    for req in reqs:
        mapping[req.project_name] = req

    return mapping


required_packages = requirements_of_file(Path(pkg_resources.resource_filename(
    "trulens_eval", "requirements.txt"
)))
optional_packages = requirements_of_file(Path(pkg_resources.resource_filename(
    "trulens_eval", "requirements.optional.txt"
)))

all_packages = {**required_packages, **optional_packages}

def format_missing_imports(
    packages: Union[str, Sequence[str]],
    purpose: str,
    throw: Union[bool, Exception] = False
) -> str:
    
    """
    Format a message indicating the given `packages` are required for `purpose`.
    Throws an `ImportError` with the formatted message if `throw` flag is set.
    If `throw` is already an exception, throws that instead after printing the
    message.
    """

    if isinstance(packages, str):
        packages = [packages]

    requirements = []
    for pkg in packages:
        if pkg in all_packages:
            requirements.append(str(all_packages[pkg]))
        else:
            print(f"WARNING: package {pkg} not present in requirements.")
            requirements.append(pkg)

    msg = (
        f"{','.join(packages)} {'packages are' if len(packages) > 1 else 'package is'} required for {purpose}. "
        f"You should be able to install {'them' if len(packages) > 1 else 'it'} with pip:\n"
        f"  pip install '{' '.join(requirements)}'"
    )

    if isinstance(throw, Exception):
        print(msg)
        raise throw
    
    elif isinstance(throw, bool):
        if throw:
            raise ImportError(msg)
    
    return msg


REQUIREMENT_LLAMA = format_missing_imports(
    'llama-index',
    purpose="instrumenting llama_index apps"
)

REQUIREMENT_LANGCHAIN = format_missing_imports(
    'langchain',
    purpose="instrumenting langchain apps"
)

REQUIREMENT_SKLEARN = format_missing_imports(
    "scikit-learn",
    purpose="using embedding vector distances"
)

REQUIREMENT_COHERE = format_missing_imports(
    'cohere',
    purpose="using Cohere models"
)

REQUIREMENT_BEDROCK = format_missing_imports(
    ['boto3', 'botocore'],
    purpose="using Bedrock models"
)

REQUIREMENT_OPENAI = format_missing_imports(
    'openai',
    purpose="using OpenAI models"
)

REQUIREMENT_BERT_SCORE = format_missing_imports(
    "bert-score",
    purpose="measuring BERT Score"
)

REQUIREMENT_EVALUATE = format_missing_imports(
    "evaluate",
    purpose="using certain metrics"
)

class Dummy(object):
    """
    Class to pretend to be a module or some other imported object. Will raise an
    error if accessed in any way.
    """

    def __init__(self, message: str, importer=None):
        self.message = message
        self.importer = importer

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.message)

    def __getattr__(self, name):
        # If in OptionalImport context, create a new dummy for the requested
        # attribute. Otherwise raise error.

        if self.importer is not None and self.importer.importing:
            return Dummy(message=self.message, importer=self.importer)

        raise ModuleNotFoundError(self.message)


class OptionalImports(object):
    """
    Helper context manager for doing multiple imports from an optional module:

    ```python

        with OptionalImports(message='Please install llama_index first'):
            import llama_index
            from llama_index import query_engine

    ```

    The above python block will not raise any errors but once anything else
    about llama_index or query_engine gets accessed, an error is raised with the
    specified message (unless llama_index is installed of course).
    """

    def __init__(self, message: str = None):
        self.message = message
        self.importing = False
        self.imp = builtins.__import__

    def __import__(self, *args, **kwargs):
        try:
            return self.imp(*args, **kwargs)

        except ModuleNotFoundError as e:
            # Check if the import error was from an import in trulens_eval as
            # otherwise we don't want to intercept the error as some modules
            # rely on import failures for various things.
            module_name = inspect.currentframe().f_back.f_globals["__name__"]
            if not module_name.startswith("trulens_eval"):
                raise e
            logger.debug(f"Could not import {args[0]}.")
            return Dummy(message=self.message, importer=self)

    def __enter__(self):
        builtins.__import__ = self.__import__
        self.importing = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.importing = False
        builtins.__import__ = self.imp

        if exc_value is None:
            return None

        print(self.message)
        # Will re-raise exception unless True is returned.
        return None
