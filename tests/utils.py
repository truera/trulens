"""Testing utilities."""

import builtins
from collections import namedtuple
import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import (
    ForwardRef,
    Iterable,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    get_args,
)

from trulens.core.utils import python as python_utils

Member = namedtuple("Member", ["obj", "name", "qualname", "val", "typ"])
"""Class/module member information.

Contents:

    - obj: object or class owning the member

    - name: name of the member

    - qualname: fully qualified name of the member

    - val: member's value

    - typ: member's type
"""


def type_str(typ: Union[type, ForwardRef, TypeVar]) -> str:
    """Render the type/type constructor with all referenced types using fully
    qualified names.

    Can also handle ForwardRefs as long as they have been evaluated already,
    ellipses, and TypeVars.
    """

    if isinstance(typ, ForwardRef):
        if not typ.__forward_evaluated__:
            raise ValueError(
                f"Not evaluated ForwardRefs are not supported: {typ}."
            )

        return type_str(typ.__forward_value__)

    if isinstance(typ, TypeVar):
        return typ.__name__

    if typ is ...:
        return "..."

    parts = []

    if hasattr(typ, "__module__"):
        parts.append(typ.__module__)

    if hasattr(typ, "__qualname__"):
        parts.append(typ.__qualname__)
    else:
        if hasattr(typ, "__name__"):
            parts.append(typ.__name__)
        else:
            parts.append(str(typ))

    ret = ".".join(parts)

    if hasattr(typ, "__args__"):
        ret += "[" + ", ".join(type_str(arg) for arg in get_args(typ)) + "]"

    return ret


def get_module_names_of_path(path: str, prefix: str = "") -> Iterable[str]:
    """Get all modules in the given path.

    Args:
        mod: The base module over which to look.
    """

    for modinfo in pkgutil.walk_packages(path, prefix=prefix):
        yield modinfo.name


def get_submodule_names(mod: ModuleType) -> Iterable[str]:
    """Get all modules in the given path.

    Args:
        mod: The base module over which to look.
    """
    if isinstance(mod, str):
        try:
            mod = importlib.import_module(mod)
        except Exception:
            return

    path = mod.__path__

    yield mod.__name__

    for modname in get_module_names_of_path(path, prefix=mod.__name__ + "."):
        if modname.endswith("_bundle"):
            # Skip this as it is not a real module/package.
            # TODO: figure out how to prevent it from being installed to begin with.
            continue

        yield modname


def get_module_exports(mod: Union[str, ModuleType]) -> Iterable[Member]:
    """Get all exports of the given module, i.e. members that are listed in
    `__all__`.

    Args:
        mod: The module or its name.

    Returns:
        Iterable of Member namedtuples.
    """

    if isinstance(mod, str):
        try:
            mod = importlib.import_module(mod)
        except Exception:
            return

    if inspect.getattr_static(mod, "__all__", None) is None:
        return

    for name in mod.__all__:
        val = inspect.getattr_static(mod, name)
        qualname = mod.__name__ + "." + name
        yield Member(mod, name=name, qualname=qualname, val=val, typ=type(val))


def _isdefinedin(val, mod: Union[ModuleType, None]):
    """Check if a value has a defining module and if it is the given module.

    Args:
        val: The value to check.

        mod: The module to check against. If `None`, will return True if `val`
            is not defined anywhere (is a builtin type).
    """

    try:
        return mod is inspect.getmodule(val)

    except Exception:
        return mod is None


def get_module_definitions(mod: Union[str, ModuleType]) -> Iterable[Member]:
    """Get all members defined in the given module, i.e. members that are not
    aliases with definitions somewhere else.

    A limitation is that basic types like int, str, etc. are not produced as
    their location of definition cannot be easily checked.

    Args:
        mod: The module or its name.

    Returns:
        Iterable of namedtuples (mod, qualname, val) where qualname is the
            fully qualified name of the member and val is the value of the
            member.
    """

    if isinstance(mod, str):
        try:
            mod = importlib.import_module(mod)
        except Exception:
            return

    for item, val in python_utils.getmembers_static(mod):
        # Check for aliases: classes/modules defined somewhere outside of
        # mod. Note that this cannot check for aliasing of basic python
        # values which are not references.
        if not _isdefinedin(val, mod):
            continue

        qualname = mod.__name__ + "." + item
        yield Member(mod, name=item, qualname=qualname, val=val, typ=type(val))


def get_definitions(mod: Union[ModuleType, str]) -> Iterable[Member]:
    """Get all definitions in the module.

    Definitions are members that are not aliases with definitions somewhere
    else. A limitation is that basic types like int, str, etc. are not produced
    as their location of definition cannot be easily checked.

    Args:
        mod: Module whose definitions we want to get. Includes submodules.


    Returns:
        Iterable of namedtuples (modname, qualname, val) where modname is the
            module name, qualname is the fully qualified name of the member and
            val is the value of the member.
    """

    if isinstance(mod, str):
        try:
            mod = importlib.import_module(mod)
        except Exception:
            return

    for modname in get_submodule_names(mod):
        yield from get_module_definitions(modname)


def get_exports(mod: Union[ModuleType, str]) -> Iterable[Member]:
    """Get all exports in the given path.

    Exports are values whose names are in the `__all__` special variable of
    their owning module.

    Args:
        path: The path to search for exports. Can be iterable of paths in case
            of namespace package paths.

        prefix: If given, only exports from modules that match this
            prefix are returned.

    Returns:
        Iterable of namedtuples (modname, qualname, val) where modname is the
            module name, qualname is the fully qualified name of the member and
            val is the value of the member.
    """

    if isinstance(mod, str):
        try:
            mod = importlib.import_module(mod)
        except Exception:
            return

    for modname in get_submodule_names(mod):
        if inspect.getattr_static(mod, "__all__", None) is None:
            continue

        for name in mod.__all__:
            val = getattr(mod, name)
            qualname = modname + "." + name
            yield Member(
                mod, name=name, qualname=qualname, val=val, typ=type(val)
            )


Members = namedtuple(
    "Members",
    [
        "obj",
        "version",
        "exports",
        "definitions",
        "access_publics",
        "access_friends",
        "access_privates",
        "api_highs",
        "api_lows",
    ],
)
"""Members of a module or a class organized into several groups.

_Exports_ of are members listed in __all__. Exports does not apply to classes.

_Definitions_ are non-base type members whose value comes from the module or
class itself (as opposed to being an alias for a
    definition in some other module).

The types of _access_:

    - _private_: Members whose names start with two underscores but does not end
        with two underscores.

    - _friend_: Members whose names start with one underscore.

    - _public_: Members whose names do not start with an underscore.

The levels of API of modules:

    - _high-level_: Public members that are exported. It is an error to have an
      exported member that is not public.

    - _low-level_: Public members that are either defined in the given module or
      are base types (str, int, ...).

The API level of class members is the API level of the class itself.
"""


def get_class_members(class_: type, class_api_level: str = "low") -> Members:
    """Get all members of a class.

    Args:
        cls: The class.

    Returns:
        Iterable of Member namedtuples.
    """

    definitions: List[Member] = []
    exports: List[Member] = []
    publics: List[Member] = []
    friends: List[Member] = []
    privates: List[Member] = []

    highs: List[Member] = []
    lows: List[Member] = []

    static_members = [
        (k, v, type(v)) for k, v in python_utils.getmembers_static(class_)
    ]

    slot_members = []
    if hasattr(class_, "__slots__"):
        slot_members = [
            (name, v := getattr(class_, name), type(v))
            for name in class_.__slots__
        ]

    # Cannot get attributes of pydantic models in the above ways so we have
    # special handling for them here:
    fields_members = []
    if hasattr(class_, "model_fields"):  # pydantic.BaseModel (v2)
        fields_members = [
            (name, field.default, field.annotation)
            for name, field in class_.model_fields.items()
        ]
    elif hasattr(class_, "__fields__"):  # pydantic.v1.BaseModel
        fields_members = [
            (name, field.default, field.annotation)
            for name, field in class_.__fields__.items()
        ]

    for name, val, typ in static_members + slot_members + fields_members:
        qualname = class_.__module__ + "." + class_.__qualname__ + "." + name
        member = Member(
            class_.__module__, name=name, qualname=qualname, val=val, typ=typ
        )

        is_def = False
        is_public = False

        if any(
            ((not base.__name__.startswith("trulens")) and hasattr(base, name))
            for base in class_.__bases__
        ):
            # Skip anything that is a member of non trulens_eval bases.
            continue

        if name.startswith("__") and not name.endswith("__"):
            group = privates

        elif name.startswith("_"):
            group = friends

        else:
            is_public = True
            group = publics

        if not any(
            (base.__name__.startswith("trulens")) and hasattr(base, name)
            for base in class_.__bases__
        ):
            # View the class-equivalent of a definitions as members of a class
            # that are not members of its bases.
            is_def = True
            definitions.append(member)

        # There is no class-equivalent notion of exports.
        group.append(member)

        if is_public and is_def:
            if class_api_level == "high":
                highs.append(member)
            else:
                lows.append(member)

    return Members(
        obj=class_,
        version=None,
        exports=exports,
        definitions=definitions,
        access_publics=publics,
        access_friends=friends,
        access_privates=privates,
        api_highs=highs,
        api_lows=lows,
    )


def get_module_members(mod: Union[str, ModuleType]) -> Optional[Members]:
    """Get all members of a module organized into exports, definitions;
    three types of access: public, friends, privates; and two levels of API:
    high-level and low-level.

    Args:
        mod: The module or its name.

    Returns:
        A namedtuples (modname, exports, definitions, access_publics,
            access_friends, access_privates, api_highs, api_lows) where the all
            but the first are iterables of namedtuples (modname, qualname, val)
            where qualname is the fully qualified name of the member and val is
            the value of the member.
    """

    if isinstance(mod, str):
        try:
            mod = importlib.import_module(mod)
        except Exception:
            return None

    definitions: List[Member] = []
    exports: List[Member] = []
    publics: List[Member] = []
    friends: List[Member] = []
    privates: List[Member] = []

    highs = []
    lows = []

    export_set: Set[str] = set(inspect.getattr_static(mod, "__all__", []))

    classes = set()

    for name, val in python_utils.getmembers_static(mod):
        qualname = mod.__name__ + "." + name
        member = Member(
            mod, name=name, qualname=qualname, val=val, typ=type(val)
        )

        is_def = False
        is_public = False
        is_export = False
        is_base = False

        if _isdefinedin(val, mod):
            is_def = True
            definitions.append(member)

            if inspect.isclass(val):
                classes.add(val)

        if _isdefinedin(val, None):
            # Is a base type that does not define a owner module.
            is_base = True

        if name in export_set:
            is_export = True
            exports.append(member)

        if (
            name.startswith("__")
            and name.endswith("__")
            and not hasattr(builtins, name)
            and name
            not in [
                "__file__",
                "__cached__",
                "__builtins__",
                "__warningregistry__",
            ]
        ):
            # Checking if name is in builtins filters out standard module
            # attributes like __package__. A few other standard attributes are
            # filtered out too.
            is_public = True
            group = publics

        elif name.startswith("__") and not name.endswith("__"):
            group = privates

        elif name.startswith("_"):
            group = friends

        else:
            is_public = True
            group = publics

        group.append(member)

        if (not is_public) and (name in export_set):
            raise ValueError(
                f"Member {qualname} is both exported and not public."
            )

        if is_public:
            if is_export:
                highs.append(member)
            elif is_def or is_base:
                lows.append(member)

    return Members(
        obj=mod,
        version=None if not hasattr(mod, "__version__") else mod.__version__,
        exports=exports,
        definitions=definitions,
        access_publics=publics,
        access_friends=friends,
        access_privates=privates,
        api_highs=highs,
        api_lows=lows,
    )
