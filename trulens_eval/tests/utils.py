"""Testing utilities."""

from collections import namedtuple
import importlib
import inspect
from pathlib import Path
import pkgutil
import re
from types import ModuleType
from typing import Iterable, Optional, Set, Union


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
                path / modinfo.name, matching=matching
            ):
                submodqualname = modinfo.name + "." + submod

                if matching is not None and not matching.fullmatch(
                    submodqualname
                ):
                    continue

                yield submodqualname


_Member = namedtuple("_Member", ["mod", "qualname", "val", "typ"])


def get_module_exports(mod: Union[str, ModuleType]) -> Iterable[_Member]:
    """Get all exports of the given module, i.e. members that are listed in
    `__all__`.

    Args:
        mod: The module or its name.

    Returns:
        Iterable of namedtuples (mod, qualname, val, typ) where qualname is the
            fully qualified name of the member, val is its value, and typ is its
            type.
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
        yield _Member(mod, qualname, val, type(val))

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
    
def get_module_definitions(mod: Union[str, ModuleType]) -> Iterable[_Member]:
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
        yield _Member(mod, qualname, val, type(val))


def get_definitions(
    path: Path, matching: Optional[Union[re.Pattern, str]] = None
) -> Iterable[_Member]:
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
) -> Iterable[_Member]:
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
            yield _Member(modname, qualname, val, type(val))


_Members = namedtuple(
    "_Members",
    [
        "mod",
        "exports",
        "definitions",
        "access_publics",
        "access_friends",
        "access_privates",
        "api_highs",
        "api_lows",
    ],
)


def get_module_members(mod: Union[str, ModuleType]) -> Optional[_Members]:
    """Get all members of a module organized into exports, definitions;
    three types of access: public, friends, privates; and two levels of API:
    high-level and low-level.

    Exports are members listed in __all__. Definitions are non-base type members
    whose value comes from the module itself (as opposed to being an alias for a
    definition in some other module).

    The types of access:

    - private: Members whose names start with two underscores but does not end
        with two underscores.

    - friends: Members whose names start with one underscore.

    - public: Members whose names do not start with an underscore.

    The levels of API:

    - high-level: Public members that are exported. It is an error to have an
      exported member that is not public.

    - low-level: Public members that are either defined in the given module or
      are base types (str, int, ...).

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

    definitions = []
    exports = []
    publics = []
    friends = []
    privates = []

    highs = []
    lows = []

    export_set: Set[str] = set(inspect.getattr_static(mod, "__all__", []))

    for name, val in inspect.getmembers_static(mod):
        qualname = mod.__name__ + "." + name
        member = _Member(mod, qualname, val, type(val))

        is_def = False
        is_public = False
        is_export = False
        is_base = False

        if _isdefinedin(val, mod):
            is_def = True
            definitions.append(member)

        if _isdefinedin(val, None):
            is_base = True

        if name in export_set:
            is_export = True
            exports.append(member)

        if name.startswith("__") and name.endswith("__"):
            is_public = True
            publics.append(member)

        elif name.startswith("__"): # and not ends with
            privates.append(member)
                
        elif name.startswith("_"):
            friends.append(member)

        else:
            is_public = True
            publics.append(member)

        if (not is_public) and (name in export_set):
            raise ValueError(
                f"Member {qualname} is both exported and not public."
            )

        if is_public:
            if is_export:
                highs.append(member)
            elif is_def or is_base:
                lows.append(member)

    return _Members(
        mod,
        exports=exports,
        definitions=definitions,
        access_publics=publics,
        access_friends=friends,
        access_privates=privates,
        api_highs=highs,
        api_lows=lows,
    )
