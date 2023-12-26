"""
Utilities for importing python modules and optional importing.
"""

import builtins
from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
from pprint import PrettyPrinter
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

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

    msg = (
        f"""
{','.join(packages)} {pack_s} {is_are} required for {purpose}.
You should be able to install {it_them} with pip:

    pip install '{' '.join(requirements)}'
"""
    )

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

    return ImportErrorMessages(module_not_found=msg, import_error=msg_pinned)


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

class DummyMeta(type):
    def __new__(cls, name, bases, namespace):
        print(f"meta new: {cls} {name}")
        if name != "Dummy":
            def fake_init(self, *args, **kwargs):
                raise ImportError(name)
            namespace['__init__'] = fake_init
        
        return super().__new__(cls, name, bases, namespace)
    
    
# Try to pretend to be a type as well as an instance.
class Dummy(type):
    """
    Class to pretend to be a module or some other imported object. Will raise an
    error if accessed in any way.
    """

    def __init_subclass__(cls, /, *args, **kwargs):
        # super().__init_subclass__(**kwargs)
        print("\ninit_subclass")
        print(args)
        print(kwargs)

    @staticmethod
    def __new__(cls, *args, **kwargs):
        # If we pretended to be a type, we need to handle calls to creation of a
        # new instance of that type, and raise an ImportError here.

        #print("\nDummy.__new__")
        #print(cls)
        #print(args)
        #print(kwargs)

        if len(args) == 3 and len(kwargs) == 0:
            # A class extended a dummy type. Keep producing dummies.
            name, bases, namespace = args
            # new_attrs = dict(__classcell__ = attrs['__classcell__'])
            # fill the other attributes with dummies here?
            # new_attrs['name'] = name
            # new_attrs['importer'] = None
            # new_attrs['message'] = None
            # new_attrs['exception_class'] = ImportError
            #def fake_init(*args, **kwargs):
            #    raise ImportError(name)
            #namespace['__init_subclass__'] = fake_init
            #namespace['__init__'] = fake_init
            #namespace['__call__'] = fake_init

            dummy = None
            
            for base in bases:
                if isinstance(base, Dummy):
                    print("base is dummy")
                    dummy = base
            
            assert dummy is not None
            importer = dummy.importer
            exception_class = dummy.exception_class
            message = dummy.message

            #namespace['importer'] = importer
            #namespace['message'] = message
            #namespace['exception_class'] = exception_class
            #namespace['__init__'] = dummy.__call__
            #for k, v in namespace.items():
            #    print(f"\t{k}={v}")

            #cls, name, bases, namespace,
            obj = Dummy(
                name=name,
                importer=importer,
                message=message,
                exception_class=exception_class,
                sub_bases=bases,
                sub_cls=cls,
                # sub_namespace=namespace,
                #classcell = namespace.get('__classcell__')
            )

            obj.__class__ = Dummy

            print(f"will return {obj}")

            return obj

        #if 'importer' not in kwargs:
        #    raise ImportError(f"Dummy.__new__({cls},\n{args},\n{kwargs})")

        importer = kwargs['importer']
        name = kwargs['name']

        #if not importer.importing:
        #    raise ImportError(f"Dummy.__new__({cls},\n{args},\n{kwargs})")
        
        ns = dict()
        sub_ns = kwargs.get('sub_namespace')
        if sub_ns is not None:
            ns = sub_ns
    
        sub_bases = kwargs.get('sub_bases')
        bases = (object,)
        if sub_bases is not None:
            bases = sub_bases
        
        print("\t", bases)

        return type.__new__(cls, name, bases, ns)

    def __init__(
        self,
        name: str,
        message: str,
        exception_class: Type[Exception] = ModuleNotFoundError,
        importer = None,
        **kwargs
    ):
        print(f"Dummy.__init__({name})")

        self.name = name
        self.message = message
        self.importer = importer
        self.exception_class = exception_class
        self.__class__ = Dummy

    def __call__(self, *args, **kwargs):
        print("\ncall")
        print("\t", self)
        print("\t", args)
        print("\t", kwargs)
        #if args[0] == Dummy:
        #    self = args[0]
        raise self.exception_class(self.message)
        #else:
        #    print("call without self")
            
        """
        print(f"self={self}")
        print(f"name={self.name}")
        print(f"importer={self.importer}")
        print(f"message={self.message}")
        print(f"exception_class={self.exception_class}")
        """
        
    def __instancecheck__(self, __instance: Any) -> bool:
        return True

    @classmethod
    def __subclasshook__(cls, __subclass) -> bool:
        return True

    def __subclasscheck__(self, __subclass: type) -> bool:
        return True

    def __mro_entries__(self, bases) -> Tuple[type, ...]:
        return (object, )
                
    def __typing_subst__(self, t):
        print(self, t)
        return True

    def __getattr__(self, name):
        # If in OptionalImport context, create a new dummy for the requested
        # attribute. Otherwise raise error.

        # Pretend to be object for generic attributes.
        if hasattr(object, name):
            return getattr(object, name)

        # Prevent pydantic inspecting this object as if it were a type from
        # triggering the exception message below.
        if name in ["__pydantic_generic_metadata__",
                    "__get_pydantic_core_schema__", "__get_validators__",
                    "__get_pydantic_json_schema__", "__modify_schema__",
                    "__origin__", "__dataclass_fields__"]:
            raise AttributeError()

        # If we are still in an optional import block, continue making dummies
        # inside this dummy.
        if self.importer is not None and self.importer.importing:
            return Dummy(
                name=self.name + "." + name,
                message=self.message,
                importer=self.importer,
                exception_class=self.exception_class
            )

        print(name)

        # If we are no longer in optional imports context, raise the exception
        # with the optional package message.
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
            return Dummy(
                name=name,
                message=self.messages.module_not_found,
                importer=self,
                exception_class=ImportError
            )

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
