"""
Utilities for importing python modules and optional importing.
"""

import builtins
import inspect
import logging
from pprint import PrettyPrinter

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

llama_version = "0.8.29post1"
REQUIREMENT_LLAMA = (
    f"llama_index {llama_version} or above is required for instrumenting llama_index apps. "
    f"Please install it before use: `pip install llama_index>={llama_version}`."
)

langchain_version = "0.0.302"
REQUIREMENT_LANGCHAIN = (
    f"langchain {langchain_version} or above is required for instrumenting langchain apps. "
    f"Please install it before use: `pip install langchain>={langchain_version}`."
)

REQUIREMENT_SKLEARN = (
    f"scikit-learn is required for using embedding vector distances. "
    f"Please install it before use: `pip install scikit-learn`."
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