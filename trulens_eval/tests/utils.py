"""Testing utilities."""

import builtins
from collections import namedtuple
import importlib
import inspect
from pathlib import Path
import pkgutil
import re
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


def get_module_names(
    path: Path, matching: Optional[Union[re.Pattern, str]] = None
) -> Iterable[str]:
    """Get all module names in the given path that start with the given
    prefix.

    Args:
        path: The path to search for modules.

        matching: If given, only modules that match this regular expression are
            returned. Note that if the pattern necessitates that some submodule
            must be returned, the ancestors must also match for it to be found
            and returned. That is, if r`trulens\\.feedback\\..+` is desired, you
            need to match the ancestor `trulens` as well with e.g.
            r`trulens\\.(feedback\\.)?+`. If a string is given, it is interpreted
            as a literal non-strict prefix.

    Returns:
        Iterable of module names. These are fully qualified.
    """

    if isinstance(matching, str):
        matching = re.compile(re.escape(matching) + r".*")

    for modinfo in pkgutil.iter_modules([str(path)]):
        if matching is not None and not matching.fullmatch(modinfo.name):
            continue

        yield modinfo.name

        if modinfo.ispkg:
            for submod in get_module_names(
                path / modinfo.name,
                matching=None,  # Intentionally not passing matching here as we want to only use matching on the qualified module name, which we do below.
            ):
                submodqualname = modinfo.name + "." + submod

                if matching is not None and not matching.fullmatch(
                    submodqualname
                ):
                    continue

                yield submodqualname


Member = namedtuple("Member", ["obj", "qualname", "val", "typ"])
"""Class/module member information.

Contents:

    - obj: object or class owning the member

    - qualname: fully qualified name of the member

    - val: member's value

    - typ: member's type
"""


def type_str(typ: Union[type | ForwardRef | TypeVar]) -> str:
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

    ret = typ.__module__ + "." + typ.__qualname__

    if hasattr(typ, "__args__"):
        ret += "[" + ", ".join(type_str(arg) for arg in typ.__args__) + "]"

    return ret


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
        yield Member(mod, qualname, val, type(val))


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

    for item, val in inspect.getmembers_static(mod):
        # Check for aliases: classes/modules defined somewhere outside of
        # mod. Note that this cannot check for aliasing of basic python
        # values which are not references.
        if not _isdefinedin(val, mod):
            continue

        qualname = mod.__name__ + "." + item
        yield Member(mod, qualname, val, type(val))


def get_definitions(
    path: Path, matching: Optional[Union[re.Pattern, str]] = None
) -> Iterable[Member]:
    """Get all definitions in the given path.

    Definitions are members that are not aliases with definitions somewhere
    else. A limitation is that basic types like int, str, etc. are not produced
    as their location of definition cannot be easily checked.

    Args:
        path: The path to search for definitions.

        matching: If given, only definitions from modules that match this
            regular expression are returned. If a string is given, it is
            interpreted as a literal non-strict prefix.

    Returns:
        Iterable of namedtuples (modname, qualname, val) where modname is the
            module name, qualname is the fully qualified name of the member and
            val is the value of the member.
    """

    if isinstance(matching, str):
        matching = re.compile(re.escape(matching) + r".*")

    for modname in get_module_names(path, matching=matching):
        yield from get_module_definitions(modname)


def get_exports(
    path: Path, matching: Optional[Union[re.Pattern, str]] = None
) -> Iterable[Member]:
    """Get all exports in the given path.

    Exports are values whose names are in the `__all__` special variable of
    their owning module.

    Args:
        path: The path to search for exports.

        matching: If given, only exports from modules that match this
            regular expression are returned. If a string is given, it is
            interpreted as a literal non-strict prefix.

    Returns:
        Iterable of namedtuples (modname, qualname, val) where modname is the
            module name, qualname is the fully qualified name of the member and
            val is the value of the member.
    """

    if isinstance(matching, str):
        matching = re.compile(re.escape(matching) + r".*")

    for modname in get_module_names(path, matching=matching):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        if inspect.getattr_static(mod, "__all__", None) is None:
            continue

        for name in mod.__all__:
            val = getattr(mod, name)
            qualname = modname + "." + name
            yield Member(modname, qualname, val, type(val))


Members = namedtuple(
    "Members",
    [
        "obj",
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
        (k, v, type(v)) for k, v in inspect.getmembers_static(class_)
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
    if hasattr(class_, "__fields__"):
        fields_members = [
            (name, field.default, field.annotation)
            for name, field in class_.__fields__.items()
        ]

    for name, val, typ in static_members + slot_members + fields_members:
        qualname = class_.__module__ + "." + class_.__qualname__ + "." + name
        member = Member(class_.__module__, qualname, val, typ)

        is_def = False
        is_public = False

        if any(
            (
                (not base.__name__.startswith("trulens_eval"))
                and hasattr(base, name)
            )
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
            (base.__name__.startswith("trulens_eval")) and hasattr(base, name)
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
        class_,
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

    for name, val in inspect.getmembers_static(mod):
        qualname = mod.__name__ + "." + name
        member = Member(mod, qualname, val, type(val))

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

        elif name.startswith("__"):  # and not ends with
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
        mod,
        exports=exports,
        definitions=definitions,
        access_publics=publics,
        access_friends=friends,
        access_privates=privates,
        api_highs=highs,
        api_lows=lows,
    )
