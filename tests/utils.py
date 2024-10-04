"""Testing utilities."""

import builtins
from collections import namedtuple
import ctypes
import gc
import importlib
import inspect
import pkgutil
from queue import Queue
from types import ModuleType
from typing import (
    Any,
    Callable,
    ForwardRef,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    get_args,
)
import weakref

from tqdm.auto import tqdm
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils.text import format_size

T = TypeVar("T")

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
        if modname.endswith("._bundle") or modname.startswith(
            "trulens.dashboard.pages"
        ):
            # Skip _bundle this as it is not a real module/package.

            # Skip trulens.dashboard.pages* because importing them executes a lot of stuff.

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


def get_class_members(
    class_: type,
    class_api_level: str = "low",
    class_alias: Optional[str] = None,
    overrides_are_defs: bool = False,
) -> Members:
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

    qualbase: str = (
        class_.__module__ + "." + class_.__qualname__
        if class_alias is None
        else class_alias
    )

    for name, val, typ in static_members + slot_members + fields_members:
        qualname = qualbase + "." + name
        member = Member(
            class_.__module__, name=name, qualname=qualname, val=val, typ=typ
        )

        is_def = False
        is_public = False
        is_experimental = False

        is_experimental = name.lower().startswith("experimental")

        if is_experimental:
            # Skip experimental members for now.
            # TODO: include them as another category other than public.
            continue

        if any(
            (
                (not base.__module__.startswith("trulens"))
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

        if overrides_are_defs or not any(
            (base.__module__.startswith("trulens")) and hasattr(base, name)
            for base in class_.__bases__
        ):
            # View the class-equivalent of a definitions as members of a class
            # that are not members of its bases.

            # If alias provided, we consider everything to be a definition though.
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


def get_module_members(
    mod: Union[str, ModuleType], aliases_are_defs: bool = False
) -> Optional[Members]:
    """Get all members of a module organized into exports, definitions;
    three types of access: public, friends, privates; and two levels of API:
    high-level and low-level.

    Args:
        mod: The module or its name.

        aliases_are_defs: If True, members that are aliases for definitions in other
            modules are considered definitions here too. This is to catch
            deprecation aliases.

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
        except Exception as e:
            if mod == "trulens.core.database.migrations.env":
                # This cannot be imported except by alembic. Skip this import
                # error.
                return None

            raise ValueError(f"Could not import module {mod}.") from e

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
        is_experimental = False

        if name.lower().startswith("experimental"):
            is_experimental = True

        if aliases_are_defs or _isdefinedin(val, mod):
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
                "__path__",
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
            if (
                not is_experimental
                and name not in ["TYPE_CHECKING"]  # skip this common value
                and (len(name) > 1 or not isinstance(val, TypeVar))
            ):  # skip generic type vars
                is_public = True
                group = publics
            else:
                is_public = False
                group = privates

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


# Reference and garbage collection related utilities


def deref(obj_id: int) -> Any:
    """Dereference an object from its id.

    Will crash the interpreter if the id is invalid.
    """

    return ctypes.cast(obj_id, ctypes.py_object).value


class RefLike(Generic[T]):
    """Container for objects that try to put them in a weakref if possible."""

    def __init__(
        self,
        ref_id: Optional[int] = None,
        obj: Optional[Union[weakref.ReferenceType[T], T]] = None,
    ):
        if ref_id is None:
            assert obj is not None
            ref_id = id(obj)

        if obj is None:
            ref_or_val = deref(ref_id)
            assert ref_or_val is not None
        else:
            ref_or_val = obj

        assert ref_or_val is not None
        if gc.is_tracked(obj):
            try:
                self.ref_or_val = weakref.ref(obj)
            except Exception:
                self.ref_or_val = obj
        else:
            self.ref_or_val = obj

        self.ref_id = ref_id
        self.ref_or_val = ref_or_val

    def get(self) -> T:
        try:
            if weakref.ReferenceType.__instancecheck__(self.ref_or_val):
                return self.ref_or_val()
        except Exception:
            return None

        return self.ref_or_val


class GetReferent(serial_utils.Step):
    """A lens step which represents a GC referent which cannot be determined to
    be an item/index/attribute yet is still a referent according to the garbage
    collector.

    This is only used for debugging purposes.
    """

    ref_id: int

    def __hash__(self) -> int:
        return hash(self.ref_id)

    def get(self, obj: Any) -> Iterable[Any]:
        yield deref(self.ref_id)

    def set(self, obj: Any, val: Any) -> Any:
        raise NotImplementedError("Cannot set a reference.")

    def __repr__(self) -> str:
        val = deref(self.ref_id)
        typ = type(val).__name__
        return f"/{typ}@{self.ref_id:08x}"


def find_path(source_id: int, target_id: int) -> Optional[serial_utils.Lens]:
    """Find the reference path between two python objects by their id.

    Returns None if a path is not found. Tries to describe the path in terms of
    dictionary or array lookups but may fall back to referents which only
    indicate that an object is linked to another as per python's GC but how the
    link is represented is not known to us.
    """

    visited: Set[int] = set()
    queue: Queue[Tuple[serial_utils.Lens, List[RefLike]]] = Queue()

    source_ref = RefLike(ref_id=source_id)
    queue.put((serial_utils.Lens(), [source_ref]))

    prog = tqdm(total=len(gc.get_objects()))

    def skip_val(val):
        if val is None:
            return True
        if not gc.is_tracked(val):
            return True
        if gc.is_finalized(val):
            return True
        try:
            if weakref.CallableProxyType.__instancecheck__(val):
                return True
            if weakref.ReferenceType.__instancecheck__(val):
                return True
            if weakref.ProxyType.__instancecheck__(val):
                return True
        except Exception:
            return False
        if id(val) in visited:
            return True

        return False

    biggest_len = 0

    while not queue.empty():
        lens, path = queue.get()

        prog.update(len(visited) - prog.n)

        if len(path) > biggest_len:
            biggest_len = len(path)

            if len(path) < 15:
                prog.set_description_str(str(lens))
            else:
                prog.set_description_str(f"lens with {len(path)} steps")
            prog.set_postfix_str(
                format_size(len(visited)) + " reference(s) visited"
            )

        final_ref = path[-1]
        if final_ref.ref_id == target_id:
            return lens

        final = final_ref.get()

        if isinstance(final, dict):
            for key, value in list(final.items()):
                if skip_val(value):
                    continue

                value_ref = RefLike(obj=value)
                visited.add(value_ref.ref_id)

                if isinstance(key, str):
                    queue.put((
                        lens + serial_utils.GetItem(item=key),
                        path + [value_ref],
                    ))
                else:
                    queue.put((
                        lens + GetReferent(ref_id=value_ref.ref_id),
                        path + [value_ref],
                    ))

        elif isinstance(final, (list, tuple)):
            for index, value in enumerate(final):
                if skip_val(value):
                    continue

                value_ref = RefLike(obj=value)
                visited.add(value_ref.ref_id)

                queue.put((
                    lens + serial_utils.GetIndex(index=index),
                    path + [value_ref],
                ))

                del value

        else:
            for value in gc.get_referents(final):
                if skip_val(value):
                    continue

                value_ref = RefLike(obj=value)
                visited.add(value_ref.ref_id)

                queue.put((
                    lens + GetReferent(ref_id=value_ref.ref_id),
                    path + [value_ref],
                ))

                del value

    return None


def print_referent_lens(lens: Optional[serial_utils.Lens], origin) -> None:
    """Print a lens that may contain GetReferent steps.

    Prints part of the referent string representation.
    """

    if lens is None:
        print("no path")

    print(lens)

    obj = origin

    for step in lens.path:
        obj = step.get_sole_item(obj)

        obj_ident = (
            type(obj).__module__ + "." + python_utils.class_name(type(obj))
        )

        if isinstance(obj, Callable):
            obj_ident += (
                " = " + obj.__module__ + "." + python_utils.callable_name(obj)
            )

        if isinstance(step, GetReferent):
            print(
                "  ",
                type(step).__name__,
                repr(step),
                obj_ident,
                str(obj)[0:1024],
            )
        else:
            print("  ", type(step).__name__, repr(step), obj_ident)
