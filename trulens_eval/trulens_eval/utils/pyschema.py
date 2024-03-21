"""
# Serialization of Python objects

In order to serialize (and optionally deserialize) python entities while still
being able to inspect them in their serialized form, we employ several storage
classes that mimic basic python entities:

| Serializable representation | Python entity |
| --- | --- |
| Class    | (python) class    |
| Module   | (python) module   |
| Obj      | (python) object   |
| Function | (python) function |
| Method   | (python) method   |

"""

from __future__ import annotations

import importlib
import inspect
import logging
from pprint import PrettyPrinter
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import warnings

import pydantic

from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import safe_issubclass
from trulens_eval.utils.serial import SerialModel

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

# Field/key name used to indicate a circular reference in dictified objects.
CIRCLE = "__tru_circular_reference"

# Field/key name used to indicate an exception in property retrieval (properties
# execute code in property.fget).
ERROR = "__tru_property_error"

# Key for indicating non-serialized objects in json dumps.
NOSERIO = "__tru_non_serialized_object"


def is_noserio(obj):
    """
    Determines whether the given json object represents some non-serializable
    object. See `noserio`.
    """
    return isinstance(obj, dict) and NOSERIO in obj


def noserio(obj, **extra: Dict) -> dict:
    """
    Create a json structure to represent a non-serializable object. Any
    additional keyword arguments are included.
    """

    inner = Obj.of_object(obj).model_dump()
    inner.update(extra)

    if isinstance(obj, Sequence):
        inner['len'] = len(obj)

    return {NOSERIO: inner}


# TODO: rename as functionality optionally produces JSONLike .
def safe_getattr(obj: Any, k: str, get_prop: bool = True) -> Any:
    """
    Try to get the attribute `k` of the given object. This may evaluate some
    code if the attribute is a property and may fail. In that case, an dict
    indicating so is returned.

    If `get_prop` is False, will not return contents of properties (will raise
    `ValueException`).
    """

    v = inspect.getattr_static(obj, k)

    is_prop = False
    try:
        # OpenAI version 1 classes may cause this isinstance test to raise an
        # exception.
        is_prop = isinstance(v, property)
    except Exception as e:
        return {ERROR: Obj.of_object(e)}

    if is_prop:
        if not get_prop:
            raise ValueError(f"{k} is a property")

        try:
            v = v.fget(obj)
            return v

        except Exception as e:
            return {ERROR: Obj.of_object(e)}
    else:
        return v


def clean_attributes(obj, include_props: bool = False) -> Dict[str, Any]:
    """
    Determine which attributes of the given object should be enumerated for
    storage and/or display in UI. Returns a dict of those attributes and their
    values.

    For enumerating contents of objects that do not support utility classes like
    pydantic, we use this method to guess what should be enumerated when
    serializing/displaying.

    If `include_props` is True, will produce attributes which are properties;
    otherwise those will be excluded. 
    """

    keys = dir(obj)

    ret = {}

    for k in keys:
        if k.startswith("__"):
            # These are typically very internal components not meant to be
            # exposed beyond immediate definitions. Ignoring these.
            continue

        if include_props and k.startswith("_") and k[1:] in keys:
            # Objects often have properties named `name` with their values
            # coming from `_name`. This check lets avoid including both the
            # property and the value.
            continue

        try:
            v = safe_getattr(obj, k, get_prop=include_props)
            ret[k] = v
        except Exception as e:
            logger.debug(str(e))

    return ret


class Module(SerialModel):
    package_name: Optional[str] = None  # some modules are not in a package
    module_name: str

    def of_module(mod: ModuleType, loadable: bool = False) -> 'Module':
        if loadable and mod.__name__ == "__main__":
            # running in notebook
            raise ImportError(f"Module {mod} is not importable.")

        return Module(package_name=mod.__package__, module_name=mod.__name__)

    def of_module_name(module_name: str, loadable: bool = False) -> 'Module':
        if loadable and module_name == "__main__":
            # running in notebook
            raise ImportError(f"Module {module_name} is not importable.")

        try:
            mod = importlib.import_module(module_name)
            package_name = mod.__package__
            return Module(package_name=package_name, module_name=module_name)

        except ImportError:
            return Module(package_name=None, module_name=module_name)

    def load(self) -> ModuleType:
        return importlib.import_module(
            self.module_name, package=self.package_name
        )


class Class(SerialModel):
    """
    A python class. Should be enough to deserialize the constructor. Also
    includes bases so that we can query subtyping relationships without
    deserializing the class first.
    """

    name: str

    module: Module

    bases: Optional[Sequence[Class]] = None

    def __repr__(self):
        return self.module.module_name + "." + self.name

    def __str__(self):
        return f"{self.name}({self.module.module_name if self.module is not None else 'no module'})"

    def base_class(self) -> 'Class':
        """
        Get the deepest base class in the same module as this class.
        """
        if self.bases is None:
            return self

        module_name = self.module.module_name

        for base in self.bases[::-1]:
            if base.module.module_name == module_name:
                return base

        return self

    def _check_importable(self):
        try:
            cls = self.load()
        except Exception as e:
            print(e)
            raise ImportError(
                f"Class {self} is not importable. "
                "If you are defining custom feedback function implementations, make sure they can be imported by python scripts. "
                "If you defined a function in a notebook, it will not be importable."
            )

    @staticmethod
    def of_class(
        cls: type, with_bases: bool = False, loadable: bool = False
    ) -> 'Class':
        ret = Class(
            name=cls.__name__,
            module=Module.of_module_name(object_module(cls), loadable=loadable),
            bases=list(map(lambda base: Class.of_class(cls=base), cls.__mro__))
            if with_bases else None
        )

        if loadable:
            if "<locals>" in repr(
                    cls):  # TODO: figure out a better way to check this
                raise ImportError(f"Class {cls} is not globally importable.")

            ret._check_importable()

        return ret

    @staticmethod
    def of_object(
        obj: object, with_bases: bool = False, loadable: bool = False
    ):
        return Class.of_class(
            cls=obj.__class__, with_bases=with_bases, loadable=loadable
        )

    @staticmethod
    def of_class_info(json: dict):
        assert CLASS_INFO in json, "Class info not in json."
        return Class.model_validate(json[CLASS_INFO])

    def load(self) -> type:  # class
        try:
            mod = self.module.load()
            return getattr(mod, self.name)

        except Exception as e:
            raise RuntimeError(f"Could not load class {self} because {e}.")

    def noserio_issubclass(self, class_name: str, module_name: str):
        bases = self.bases

        assert bases is not None, "Cannot do subclass check without bases. Serialize me with `Class.of_class(with_bases=True ...)`."

        for base in bases:
            if base.name == class_name and base.module.module_name == module_name:
                return True

        return False


Class.model_rebuild()


# inspect.signature does not work on builtin type constructors but they are used
# like this method. Use it to create a signature of a builtin constructor.
def builtin_init_dummy(self, *args, **kwargs):
    pass


builtin_init_sig = inspect.signature(builtin_init_dummy)


def _safe_init_sig(cls):
    """
    Get the signature of the constructor method of the given class `cls`. If it is
    a builtin class, this typically raises an exeception in which case we return
    a generic signature that seems typical of builtin constructors.
    """

    try:
        return inspect.signature(cls)
    except Exception as e:
        return builtin_init_sig


class Obj(SerialModel):
    # TODO: refactor this into something like WithClassInfo, perhaps
    # WithObjectInfo, and store required constructor inputs as attributes with
    # potentially a placeholder for additional arguments that are not
    # attributes, under a special key like "__tru_object_info".
    """
    An object that may or may not be loadable from its serialized form. Do not
    use for base types that don't have a class. Loadable if `init_bindings` is
    not None.
    """

    cls: Class

    # From id(obj), identifies memory location of a python object. Use this for
    # handling loops in JSON objects.
    id: int

    # Loadable
    init_bindings: Optional[Bindings] = None

    @staticmethod
    def of_object(
        obj: object, cls: Optional[type] = None, loadable: bool = False
    ) -> Obj:
        if cls is None:
            cls = obj.__class__

        bindings = None

        if loadable:
            # Constructor arguments for some common types.
            if isinstance(obj, pydantic.BaseModel):
                # NOTE: avoids circular import:
                from trulens_eval.utils.json import jsonify

                init_args = ()
                init_kwargs = obj.model_dump()
                # init_kwargs = jsonify(obj)

            elif isinstance(obj, Exception):
                init_args = obj.args
                init_kwargs = {}

            else:
                # For unknown types, check if the constructor for cls expect
                # arguments and fail if so as we don't know where to get them. If
                # there are none, create empty init bindings.

                sig = _safe_init_sig(cls)
                if len(sig.parameters) > 0:
                    raise RuntimeError(
                        f"Do not know how to get constructor arguments for object of type {cls.__name__}. "
                        f"If you are defining a custom feedback function, define its implementation as a function or a method of a Provider subclass."
                    )

                init_args = ()
                init_kwargs = {}

            # TODO: dataclasses
            # TODO: dataclasses_json

            # NOTE: Something related to pydantic models incorrectly sets signature
            # of cls so we need to check cls.__call__ instead.
            # TODO: app serialization
            #if isinstance(cls, type):
            #    sig = _safe_init_sig(cls)
            #else:

            sig = _safe_init_sig(cls.__call__)

            b = sig.bind(*init_args, **init_kwargs)
            bindings = Bindings.of_bound_arguments(b)

        cls_serial = Class.of_class(cls)

        if loadable:
            cls_serial._check_importable()

        return Obj(cls=cls_serial, id=id(obj), init_bindings=bindings)

    def load(self) -> object:
        if self.init_bindings is None:
            raise RuntimeError(
                "Cannot load object unless `init_bindings` is set."
            )

        cls = self.cls.load()

        if issubclass(cls, pydantic.BaseModel):
            # For pydantic Models, use model_validate to reconstruct object:
            return cls.model_validate(self.init_bindings.kwargs)

        else:

            sig = _safe_init_sig(cls)

            if CLASS_INFO in sig.parameters and CLASS_INFO not in self.init_bindings.kwargs:
                extra_kwargs = {CLASS_INFO: self.cls}
            else:
                extra_kwargs = {}

            try:
                bindings = self.init_bindings.load(
                    sig, extra_kwargs=extra_kwargs
                )

            except Exception as e:
                msg = f"Error binding constructor args for object:\n"
                msg += str(e) + "\n"
                msg += f"\tobj={self}\n"
                msg += f"\targs={self.init_bindings.args}\n"
                msg += f"\tkwargs={self.init_bindings.kwargs}\n"
                raise type(e)(msg)

            return cls(*bindings.args, **bindings.kwargs)


class Bindings(SerialModel):
    args: Tuple
    kwargs: Dict[str, Any]

    @staticmethod
    def of_bound_arguments(b: inspect.BoundArguments) -> Bindings:
        return Bindings(args=b.args, kwargs=b.kwargs)

    def _handle_providers_load(self):
        # HACK004: A Hack: reason below
        # This was introduced with the feedback functions Groundedness and GroundTruthAgreement.

        # `summarize_provider` explanation:
        ## The use case is a Serialized feedback, with attribute that needs instantiation
        ## But should not be a user supplied input kwarg.
        # `groundedness_provider` and `provider` explanation
        ## The rest of the providers need to be instantiated, but are currently in circular dependency if done from util.py

        if 'summarize_provider' in self.kwargs:
            del self.kwargs['summarize_provider']
        if 'groundedness_provider' in self.kwargs:
            del self.kwargs['groundedness_provider']
        if 'provider' in self.kwargs:
            del self.kwargs['provider']

    def load(self, sig: inspect.Signature, extra_args=(), extra_kwargs={}):

        # Disabling this hack as we now have different providers that may need
        # to be selected from (i.e. OpenAI vs AzureOpenAI).

        # self._handle_providers_load()

        return sig.bind(
            *(self.args + extra_args), **self.kwargs, **extra_kwargs
        )


class FunctionOrMethod(SerialModel):

    @classmethod
    def model_validate(cls, obj, **kwargs):
        if isinstance(obj, Dict):
            if 'obj' in obj:
                return super(cls, Method).model_validate(obj=obj, **kwargs)
            elif 'cls' in obj:
                return super(cls, Function).model_validate(obj=obj, **kwargs)
            else:
                raise ValueError(
                    f"Cannot tell what type of callable this encodes: {obj}"
                )
        else:
            return super().model_validate(obj)

    @staticmethod
    def of_callable(c: Callable, loadable: bool = False) -> 'FunctionOrMethod':
        """
        Serialize the given callable. If `loadable` is set, tries to add enough
        info for the callable to be deserialized.
        """

        if inspect.ismethod(c):
            self = c.__self__
            return Method.of_method(c, obj=self, loadable=loadable)

        else:

            return Function.of_function(c, loadable=loadable)

    def load(self) -> Callable:
        raise NotImplementedError()


class Method(FunctionOrMethod):
    """
    A python method. A method belongs to some class in some module and must have
    a pre-bound self object. The location of the method is encoded in `obj`
    alongside self. If obj is Obj with init_bindings, this method should be
    deserializable.
    """

    obj: Obj
    name: str

    @staticmethod
    def of_method(
        meth: Callable,
        cls: Optional[type] = None,
        obj: Optional[object] = None,
        loadable: bool = False
    ) -> 'Method':
        if obj is None:
            assert inspect.ismethod(
                meth
            ), f"Expected a method (maybe it is a function?): {meth}"
            obj = meth.__self__

        if cls is None:
            if isinstance(cls, type):
                # classmethod, self is a type
                cls = obj
            else:
                # normal method, self is instance of cls
                cls = obj.__class__

        obj_model = Obj.of_object(obj, cls=cls, loadable=loadable)

        return Method(obj=obj_model, name=meth.__name__)

    def load(self) -> Callable:
        obj = self.obj.load()
        return getattr(obj, self.name)


def object_module(obj):
    if safe_hasattr(obj, "__module__"):
        return getattr(obj, "__module__")
    else:
        return "builtins"


class Function(FunctionOrMethod):
    """
    A python function. Could be a static method inside a class (not instance of
    the class).
    """

    module: Module

    # For static methods in a class which we view as functions, not yet
    # supported:
    cls: Optional[Class]

    name: str

    @staticmethod
    def of_function(
        func: Callable,
        module: Optional[ModuleType] = None,
        cls: Optional[type] = None,
        loadable: bool = False
    ) -> 'Function':  # actually: class

        if module is None:
            module = Module.of_module_name(
                object_module(func), loadable=loadable
            )

        if cls is not None:
            cls = Class.of_class(cls, loadable=loadable)

        return Function(cls=cls, module=module, name=func.__name__)

    def load(self) -> Callable:
        if self.cls is not None:
            # TODO: static/class methods work in progress

            cls = self.cls.load()  # does not create object instance
            return getattr(cls, self.name)  # lookup static/class method

        else:
            mod = self.module.load()
            try:
                return getattr(mod, self.name)  # function not inside a class
            except Exception:
                raise ImportError(
                    f"Function {self} is not importable. "
                    "If you are defining custom feedback function implementations, make sure they can be imported by python scripts. "
                    "If you defined a function in a notebook, it will not be importable."
                )


# Key of structure where class information is stored.
CLASS_INFO = "tru_class_info"


class WithClassInfo(pydantic.BaseModel):
    """Mixin to track class information to aid in querying serialized components
    without having to load them.
    """

    tru_class_info: Class
    """Class information of this pydantic object for use in deserialization.
    
    Using this odd key to not pollute attribute names in whatever class we mix
    this into. Should be the same as CLASS_INFO.
    """

    # NOTE(piotrm): HACK005: for some reason, model_validate is not called for
    # nested models but the method decorated as such below is called. We use
    # this to load an object which includes our class information instead of
    # using pydantic for this loading as it would always load the object as per
    # its declared field. For example, `Provider` includes `endpoint: Endpoint`
    # but we want to load one of the `Endpoint` subclasses. We add the subclass
    # information using `WithClassInfo` meaning we can then use this method
    # below to load the subclass. Pydantic would only give us `Endpoint`, the
    # parent class.
    @pydantic.model_validator(mode='before')
    @staticmethod
    def load(obj, *args, **kwargs):
        """Deserialize/load this object using the class information in
        tru_class_info to lookup the actual class that will do the deserialization."""

        if not isinstance(obj, dict):
            return obj

        if CLASS_INFO not in obj:
            raise ValueError("No class info present in object.")

        clsinfo = Class.model_validate(obj[CLASS_INFO])
        try:
            # If class cannot be loaded, usually because it is not importable,
            # return obj as is.
            cls = clsinfo.load()
        except RuntimeError:
            return obj

        # TODO: We shouldn't be doing a lot of these pydantic details but could
        # not find how to integrate with existing pydantic functionality. Please
        # figure it out.
        validated = {}

        for k, finfo in cls.model_fields.items():
            typ = finfo.annotation
            val = finfo.get_default(call_default_factory=True)

            if k in obj:
                val = obj[k]

            try:
                if (isinstance(val, dict)) and (CLASS_INFO in val) \
                and inspect.isclass(typ) and safe_issubclass(typ, WithClassInfo):
                    subcls = Class.model_validate(val[CLASS_INFO]).load()

                    val = subcls.model_validate(val)
            except Exception as e:
                pass

            validated[k] = val

        # Note that the rest of the validation/conversions for things which are
        # not serialized WithClassInfo will be done by pydantic after we return
        # this:
        return validated

    def __init__(
        self,
        *args,
        class_info: Optional[Class] = None,
        obj: Optional[object] = None,
        cls: Optional[type] = None,
        **kwargs
    ):

        if obj is not None:
            warnings.warn(
                "`obj` does not need to be provided to WithClassInfo any more",
                DeprecationWarning
            )

        if obj is None:
            obj = self

        if obj is not None:
            cls = type(obj)

        if class_info is None:
            assert cls is not None, "Either `class_info`, `obj` or `cls` need to be specified."
            class_info = Class.of_class(cls, with_bases=True)

        kwargs[CLASS_INFO] = class_info

        super().__init__(*args, **kwargs)

    @staticmethod
    def get_class(obj_json: Dict):
        return Class.model_validate(obj_json[CLASS_INFO]).load()

    @staticmethod
    def of_object(obj: object):
        return WithClassInfo(class_info=Class.of_class(obj.__class__))

    @staticmethod
    def of_class(cls: type):  # class
        return WithClassInfo(class_info=Class.of_class(cls))

    @classmethod
    def model_validate(cls, *args, **kwargs) -> Any:
        # Note: This is here only so we can provide a pointer and some
        # instructions to render into the docs.
        """
        Deserialized a jsonized version of the app into the instance of the
        class it was serialized from.

        Note:
            This process uses extra information stored in the jsonized object
            and handled by [WithClassInfo][trulens_eval.utils.pyschema.WithClassInfo].
        """

        return super().model_validate(*args, **kwargs)


# HACK013:
Module.model_rebuild()
Class.model_rebuild()
Obj.model_rebuild()
Bindings.model_rebuild()
FunctionOrMethod.model_rebuild()
Function.model_rebuild()
Method.model_rebuild()
WithClassInfo.model_rebuild()
