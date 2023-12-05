"""
Utilities for importing python modules and optional importing.
"""

import builtins
import inspect
import logging
from pprint import PrettyPrinter
from typing import Iterable, Sequence, Union

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

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

    msg = (
        f"{','.join(packages)} is/are required for {purpose}. "
        f"You should be able to install it/them with\n"
        f"\tpip install {' '.join(packages)}"
    )

    if isinstance(throw, Exception):
        print(msg)
        raise throw
    
    elif isinstance(throw, bool):
        if throw:
            raise ImportError(msg)
    
    return msg


llama_version = "0.8.69"
REQUIREMENT_LLAMA = format_missing_imports(
    f"llama_index>={llama_version}",
    purpose="instrumenting llama_index apps"
)

langchain_version = "0.0.335"
REQUIREMENT_LANGCHAIN = format_missing_imports(
    f"langchain>={langchain_version}",
    purpose="instrumenting langchain apps"
)

REQUIREMENT_SKLEARN = format_missing_imports(
    "scikit-learn",
    purpose="using embedding vector distances"
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
