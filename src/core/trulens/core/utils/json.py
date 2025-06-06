"""Json utilities and serialization utilities dealing with json."""

from __future__ import annotations

import dataclasses
from enum import Enum
import hashlib
import inspect
import json
import logging
from pathlib import Path
from pprint import PrettyPrinter
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
)

import pydantic
from pydantic.v1 import BaseModel as v1BaseModel
from pydantic.v1.json import ENCODERS_BY_TYPE
from pydantic.v1.json import pydantic_encoder
from trulens.core.utils import constants as constant_utils
from trulens.core.utils import imports as import_utils
from trulens.core.utils import keys as key_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils

if TYPE_CHECKING:
    from trulens.core import instruments as core_instruments

with import_utils.OptionalImports(
    messages=import_utils.format_import_errors(
        ["openai", "httpx"], purpose="instrumenting apps with OpenAI components"
    )
):
    # httpx.URL and Timeout needed for openai client.
    import httpx
    from openai import Timeout

    def encode_httpx_url(obj: httpx.URL) -> str:
        return str(obj)

    def encode_openai_timeout(obj: Timeout) -> Dict[str, Any]:
        return obj.as_dict()

    ENCODERS_BY_TYPE[httpx.URL] = encode_httpx_url
    ENCODERS_BY_TYPE[Timeout] = encode_openai_timeout

logger = logging.getLogger(__name__)
pp = PrettyPrinter()
T = TypeVar("T")


def _recursive_hash(
    value: Union[dict, list, str, int, bool, float, complex, None],
    ignore_none=False,
) -> str:
    """Hash a json-like structure. Implementation is simplified from merkle_json.

    Args:
        value (Union[dict, list, str, int, bool, float, complex, None]): The value or object to hash.
        ignore_none (bool, optional): If provided, ignore None values in the hash. Defaults to False.

    Returns:
        str: The hash of the value.
    """
    if isinstance(value, list):
        h_acc = [_recursive_hash(v, ignore_none) for v in value]
        return _recursive_hash("".join(sorted(h_acc)), ignore_none)
    elif isinstance(value, dict):
        keys = sorted(value.keys())
        acc = ""
        for k in keys:
            key_val = value[k]
            if ignore_none and key_val is None:
                continue

            acc += f"{k}:{_recursive_hash(key_val, ignore_none=ignore_none)},"
        return _recursive_hash(acc, ignore_none=ignore_none)
    else:
        return hashlib.md5(str(value).encode("utf-8")).hexdigest()


# Add encoders for some types that pydantic cannot handle but we need.


def obj_id_of_obj(obj: Dict[Any, Any], prefix="obj"):
    """
    Create an id from a json-able structure/definition. Should produce the same
    name if definition stays the same.
    """

    return f"{prefix}_hash_{_recursive_hash(obj)}"


def json_str_of_obj(
    obj: Any, *args, redact_keys: bool = False, **kwargs
) -> str:
    """
    Encode the given json object as a string.
    """

    return json.dumps(
        jsonify(obj, *args, redact_keys=redact_keys, **kwargs),
        default=json_default,
        ensure_ascii=False,
    )


def json_default(obj: Any) -> str:
    """
    Produce a representation of an object which does not have a json serializer.
    """

    # Try the encoders included with pydantic first (should handle things like
    # Datetime, and our additional encoders above):
    try:
        return pydantic_encoder(obj)

    except Exception:
        # Otherwise give up and indicate a non-serialization.
        return pyschema_utils.noserio(obj)


def jsonify_for_ui(*args, **kwargs):
    """Options for jsonify common to UI displays.

    Redacts keys and hides special fields introduced by trulens.
    """

    return jsonify(*args, **kwargs, redact_keys=True, skip_specials=True)


def jsonify(
    obj: Any,
    dicted: Optional[Dict[int, serial_utils.JSON]] = None,
    instrument: Optional[core_instruments.Instrument] = None,
    skip_specials: bool = False,
    redact_keys: bool = False,
    include_excluded: bool = True,
    depth: int = 0,
    max_depth: int = 256,
) -> serial_utils.JSON:
    """Convert the given object into types that can be serialized in json.

        Args:
            obj: the object to jsonify.

            dicted: the mapping from addresses of already jsonifed objects (via id)
                to their json.

            instrument: instrumentation functions for checking whether to recur into
                components of `obj`.

            skip_specials: remove specially keyed structures from the json. These
                have keys that start with "__tru_".

            redact_keys: redact secrets from the output. Secrets are detremined by
                `keys.py:redact_value` .

            include_excluded: include fields that are annotated to be excluded by
                pydantic.

            depth: the depth of the serialization of the given object relative to
                the serialization of its container.
    `
            max_depth: the maximum depth of the serialization of the given object.
                Objects to be serialized beyond this will be serialized as
                "non-serialized object" as per `noserio`. Note that this may happen
                for some data layouts like linked lists. This value should be no
                larger than half the value set by
                [sys.setrecursionlimit][sys.setrecursionlimit].

        Returns:
            The jsonified version of the given object. Jsonified means that the the
            object is either a JSON base type, a list, or a dict with the containing
            elements of the same.
    """

    # NOTE(piotrm): We might need to do something special for the below types as
    # they are stateful if iterated. That is, they might be iterable only once
    # and iterating will break their user's interfaces.

    # These are here because we cannot iterate them or await them without
    # breaking the instrumented apps. Instead we return a placeholder value:
    if inspect.isawaitable(obj):
        return "TruLens: Cannot jsonify an awaitable object."
    if isinstance(obj, typing.Iterator):
        return "TruLens: Cannot jsonify an iterator object."
    if inspect.isgenerator(obj):
        return "TruLens: Cannot jsonify a generator object."
    if inspect.isasyncgen(obj):
        return "TruLens: Cannot jsonify an async generator object."

    # These may be necessary to prevent inadvertently stealing some items from
    # the awaitables/generators.
    if inspect.iscoroutine(obj):
        return "TruLens: Cannot jsonify a coroutine object."
    if inspect.isasyncgenfunction(obj):
        return "TruLens: Cannot jsonify an async generator function."
    if inspect.iscoroutinefunction(obj):
        return "TruLens: Cannot jsonify a coroutine function."
    if inspect.isgeneratorfunction(obj):
        return "TruLens: Cannot jsonify a generator function."

    if depth > max_depth:
        logger.debug(
            "Max depth reached for jsonify of object type '%s'.", type(obj)
        )  # careful about str(obj) in case it is recursive infinitely.

        return pyschema_utils.noserio(obj)

    skip_excluded = not include_excluded
    # Hack so that our models do not get exclude dumped which causes many
    # problems. Another variable set here so we can recurse with the original
    # include_excluded .
    if isinstance(obj, serial_utils.SerialModel):
        skip_excluded = True

    from trulens.core.instruments import Instrument

    if instrument is None:
        instrument = Instrument()

    dicted = dicted or {}

    if skip_specials:

        def recur_key(k):
            return (
                isinstance(k, serial_utils.JSON_BASES)
                and k not in constant_utils.ALL_SPECIAL_KEYS
            )

    else:

        def recur_key(k):
            return isinstance(k, serial_utils.JSON_BASES)

    if id(obj) in dicted:
        if skip_specials:
            return None

        return {constant_utils.CIRCLE: id(obj)}

    if isinstance(obj, serial_utils.JSON_BASES):
        if redact_keys and isinstance(obj, str):
            return key_utils.redact_value(obj)

        return obj

    # TODO: remove eventually
    if isinstance(obj, serial_utils.SerialBytes):
        return obj.model_dump()

    if isinstance(obj, Path):
        return str(obj)

    if type(obj) in ENCODERS_BY_TYPE:
        return ENCODERS_BY_TYPE[type(obj)](obj)

    # TODO: should we include duplicates? If so, dicted needs to be adjusted.
    new_dicted = dict(dicted)

    def recur(o):
        return jsonify(
            obj=o,
            dicted=new_dicted,
            instrument=instrument,
            skip_specials=skip_specials,
            redact_keys=redact_keys,
            include_excluded=include_excluded,
            depth=depth + 1,
            max_depth=max_depth,
        )

    content = None

    if isinstance(obj, Enum):
        content = obj.name

    elif isinstance(obj, Dict):
        forward_value = {}
        new_dicted[id(obj)] = forward_value
        forward_value.update({
            k: recur(v) for k, v in obj.items() if recur_key(k)
        })

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in forward_value.items():
                forward_value[k] = key_utils.redact_value(v=v, k=k)

        content = forward_value

    elif isinstance(obj, Sequence):
        forward_value = []
        new_dicted[id(obj)] = forward_value
        for x in (recur(v) for v in obj):
            forward_value.append(x)

        content = forward_value

    elif isinstance(obj, Set):
        forward_value = []
        new_dicted[id(obj)] = forward_value
        for x in (recur(v) for v in obj):
            forward_value.append(x)

        content = forward_value

    elif isinstance(obj, serial_utils.Lens):  # special handling of paths
        return obj.model_dump()

    elif isinstance(obj, pydantic.BaseModel):
        # Not even trying to use pydantic.dict here.

        forward_value = {}
        new_dicted[id(obj)] = forward_value
        forward_value.update({
            k: recur(python_utils.safe_getattr(obj, k))
            for k, v in type(obj).model_fields.items()
            if (not skip_excluded or not v.exclude) and recur_key(k)
        })

        for k, _ in type(obj).model_computed_fields.items():
            if recur_key(k):
                forward_value[k] = recur(python_utils.safe_getattr(obj, k))

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in forward_value.items():
                forward_value[k] = key_utils.redact_value(v=v, k=k)

        content = forward_value

    elif isinstance(obj, v1BaseModel):
        # TODO: DEDUP with pydantic.BaseModel case

        # Not even trying to use pydantic.dict here.

        forward_value = {}
        new_dicted[id(obj)] = forward_value
        forward_value.update({
            k: recur(python_utils.safe_getattr(obj, k))
            for k, v in obj.__fields__.items()
            if (not skip_excluded or not v.field_info.exclude) and recur_key(k)
        })

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in forward_value.items():
                forward_value[k] = key_utils.redact_value(v=v, k=k)

        content = forward_value

    elif dataclasses.is_dataclass(type(obj)):
        # NOTE: cannot use dataclasses.asdict as that may fail due to its use of
        # copy.deepcopy.

        forward_value = {}
        new_dicted[id(obj)] = forward_value

        forward_value.update({
            f.name: recur(python_utils.safe_getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if recur_key(f.name)
        })

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in forward_value.items():
                forward_value[k] = key_utils.redact_value(v=v, k=k)

        content = forward_value

    elif instrument.to_instrument_object(obj):
        forward_value = {}
        new_dicted[id(obj)] = forward_value

        kvs = pyschema_utils.clean_attributes(obj, include_props=True)

        # TODO(piotrm): object walks redo
        forward_value.update({
            k: recur(v)
            for k, v in kvs.items()
            if recur_key(k)
            and (
                isinstance(v, serial_utils.JSON_BASES)
                or isinstance(v, Dict)
                or isinstance(v, Sequence)
                or instrument.to_instrument_object(v)
            )
        })

        content = forward_value

    else:
        logger.debug(
            "Do not know how to jsonify an object of type '%s'.", type(obj)
        )  # careful about str(obj) in case it is recursive infinitely.

        content = pyschema_utils.noserio(obj)

    # Add class information for objects that are to be instrumented, known as
    # "components".
    if (
        not skip_specials
        and isinstance(content, dict)
        and not isinstance(obj, dict)
        and (
            instrument.to_instrument_object(obj)
            or isinstance(obj, pyschema_utils.WithClassInfo)
        )
    ):
        content[constant_utils.CLASS_INFO] = pyschema_utils.Class.of_class(
            cls=obj.__class__, with_bases=True
        ).model_dump()

    if not isinstance(obj, serial_utils.Lens) and python_utils.safe_hasattr(
        obj, "jsonify_extra"
    ):
        # Problem with Lens and similar objects: they always say they have every attribute.
        content = obj.jsonify_extra(content)

    return content
