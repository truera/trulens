"""Import utilities for required and optional imports.

Utilities for importing python modules and optional importing. This is some long line. Hopefully
this wraps automatically. 
"""

import builtins
from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
from pprint import PrettyPrinter
from typing import Any, Dict, Optional, Sequence, Type, Union

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
    Format two messages for missing optional package or bad import from an optional package. Throws
    an `ImportError` with the formatted message if `throw` flag is set. If `throw` is already an
    exception, throws that instead after printing the message.
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

    msg = (
        f"""
{','.join(packages)} {pack_s} {is_are} required for {purpose}.
You should be able to install {it_them} with pip:

    pip install {' '.join(map(lambda a: f'"{a}"', requirements))}
"""
    )

    msg_pinned = (
        f"""
You have {packs} installed but we could not import the required
components. There may be a version incompatibility. Please try installing {this_these}
exact {pack_s} with pip: 

  pip install {' '.join(map(lambda a: f'"{a}"', requirements_pinned))}

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

    return ImportErrorMessages(module_not_found=msg, import_error=msg_pinned)


REQUIREMENT_LLAMA = format_import_errors(
    'llama-index', purpose="instrumenting llama_index apps"
)

REQUIREMENT_LANGCHAIN = format_import_errors(
    'langchain', purpose="instrumenting langchain apps"
)

REQUIREMENT_PINECONE = format_import_errors(
    # package name is "pinecone-client" but module is "pinecone"
    'pinecone-client',
    purpose="running TruBot"
)

REQUIREMENT_SKLEARN = format_import_errors(
    "scikit-learn", purpose="using embedding vector distances"
)

REQUIREMENT_LITELLM = format_import_errors(
    ['litellm'], purpose="using LiteLLM models"
)

REQUIREMENT_BEDROCK = format_import_errors(
    ['boto3', 'botocore'], purpose="using Bedrock models"
)

REQUIREMENT_OPENAI = format_import_errors(
    'openai', purpose="using OpenAI models"
)

REQUIREMENT_GROUNDEDNESS = format_import_errors(
    'nltk', purpose="using some groundedness feedback functions"
)

REQUIREMENT_BERT_SCORE = format_import_errors(
    "bert-score", purpose="measuring BERT Score"
)

REQUIREMENT_EVALUATE = format_import_errors(
    "evaluate", purpose="using certain metrics"
)

REQUIREMENT_NOTEBOOK = format_import_errors(
    ["ipython", "ipywidgets"], purpose="using trulens_eval in a notebook"
)


# Try to pretend to be a type as well as an instance.
class Dummy(type, object):
    """
    Class to pretend to be a module or some other imported object. Will raise an error if accessed
    in some dynamic way. Accesses that are "static-ish" will try not to raise the exception so
    things like defining subclasses of a missing class should not raise exception. Dynamic uses are
    things like calls, use in expressions. Looking up an attribute is static-ish so we don't throw
    the error at that point but instead make more dummies.
    """

    def __new__(cls, name, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[1],
                                         dict) and "__classcell__" in args[1]:
            # (used as type)
            return type.__new__(cls, name, args[0], args[1])
        else:
            return type.__new__(cls, name, (object,), {})

    def __init__(
        self,
        name: str,
        message: str,
        exception_class: Type[Exception] = ModuleNotFoundError,
        importer=None
    ):
        self.name = name
        self.message = message
        self.importer = importer
        self.exception_class = exception_class

    def __call__(self, *args, **kwargs):
        raise self.exception_class(self.message)

    def __instancecheck__(self, __instance: Any) -> bool:
        return True

    def __subclasscheck__(self, __subclass: type) -> bool:
        return True

    def _wasused(self, *args, **kwargs):
        raise self.exception_class(self.message)

    # If someone tries to use dummy in an expression, raise our usage exception:
    __add__ = _wasused
    __sub__ = _wasused
    __mul__ = _wasused
    __truediv__ = _wasused
    __floordiv__ = _wasused
    __mod__ = _wasused
    __divmod__ = _wasused
    __pow__ = _wasused
    __lshift__ = _wasused
    __rshift__ = _wasused
    __and__ = _wasused
    __xor__ = _wasused
    __or__ = _wasused
    __radd__ = _wasused
    __rsub__ = _wasused

    def __getattr__(self, name):
        # If in OptionalImport context, create a new dummy for the requested attribute. Otherwise
        # raise error.

        # Pretend to be object for generic attributes.
        if hasattr(object, name):
            return getattr(object, name)

        # Prevent pydantic inspecting this object as if it were a type from triggering the exception
        # message below.
        if name in ["__pydantic_generic_metadata__",
                    "__get_pydantic_core_schema__", "__get_validators__",
                    "__get_pydantic_json_schema__", "__modify_schema__",
                    "__origin__", "__dataclass_fields__"]:
            raise AttributeError()

        # If we are still in an optional import block, continue making dummies
        # inside this dummy.
        if self.importer is not None and (self.importer.importing and
                                          not self.importer.fail):
            return Dummy(
                name=self.name + "." + name,
                message=self.message,
                importer=self.importer
            )

        # If we are no longer in optional imports context or context said to
        # fail anyway, raise the exception with the optional package message.
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

    def assert_installed(self, mod):
        """
        Check that the given module `mod` is not a dummy. If it is, show the
        optional requirement message.
        """
        if isinstance(mod, Dummy):
            raise ModuleNotFoundError(self.messages.module_not_found)

    def __init__(self, messages: ImportErrorMessages, fail: bool = False):
        """
        Create an optional imports context manager class. Will keep module not
        found or import errors quiet inside context unless fail is True.
        """

        self.messages = messages
        self.importing = False
        self.fail = fail
        self.imp = builtins.__import__

    def __import__(self, name, globals=None, locals=None, fromlist=(), level=0):
        try:
            mod = self.imp(name, globals, locals, fromlist, level)

            # NOTE(piotrm): commented block attempts to check module contents for required
            # attributes so we can offer a message without raising an exception later. It is
            # commented out for now it is catching some things we don't watch to catch. Need to
            # check how attributes are normally looked up in a module to fix this. Specifically, the
            # code that raises these messages: "ImportError: cannot import name ..."
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
            # HACK012: we have to step back a frame or two here to check where
            # the original import come from. We skip any frames that refer to
            # our overwritten __import__. We have to step back multiple times if
            # we accidentally nest our OptionalImport context manager.

            frame = inspect.currentframe().f_back
            while frame.f_code == self.__import__.__code__:
                frame = frame.f_back
            
            module_name = frame.f_globals["__name__"]

            logger.debug("Module not found %s in %s.", name, module_name)

            if self.fail or not module_name.startswith(trulens_name):
                raise e

            return Dummy(
                name=name,
                message=self.messages.module_not_found,
                importer=self
            )

        # NOTE(piotrm): This below seems to never be caught. It might be that a different import
        # method is being used once a module is found.
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

        # Re-raise appropriate exception.

        if isinstance(exc_value, ModuleNotFoundError):
            exc_value = ModuleNotFoundError(
                self.messages.module_not_found
            )

        elif isinstance(exc_value, ImportError):
            exc_value = ImportError(
                self.messages.import_error
            )

        else:
            raise exc_value

        # Will re-raise exception unless True is returned.
        if self.fail:
            raise exc_value

        return True
