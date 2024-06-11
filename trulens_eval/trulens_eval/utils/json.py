"""Json utilities and serialization utilities dealing with json."""

from __future__ import annotations

import dataclasses
from enum import Enum
import inspect
import json
import logging
from pathlib import Path
from pprint import PrettyPrinter
import typing
from typing import Any, Dict, Optional, Sequence, Set, TypeVar

from merkle_json import MerkleJson
import pydantic

from trulens_eval.keys import redact_value
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_OPENAI
from trulens_eval.utils.pyschema import CIRCLE
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import CLASS_INFO
from trulens_eval.utils.pyschema import clean_attributes
from trulens_eval.utils.pyschema import ERROR
from trulens_eval.utils.pyschema import NOSERIO
from trulens_eval.utils.pyschema import noserio
from trulens_eval.utils.pyschema import safe_getattr
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import JSON_BASES
from trulens_eval.utils.serial import Lens
from trulens_eval.utils.serial import SerialBytes
from trulens_eval.utils.serial import SerialModel

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")

mj = MerkleJson()

# Add encoders for some types that pydantic cannot handle but we need.

with OptionalImports(messages=REQUIREMENT_OPENAI):
    # httpx.URL needed for openai client.
    import httpx
    # Another thing we need for openai client.
    from openai import Timeout

    def encode_httpx_url(obj: httpx.URL):
        return str(obj)

    pydantic.v1.json.ENCODERS_BY_TYPE[httpx.URL] = encode_httpx_url

    def encode_openai_timeout(obj: Timeout):
        return obj.as_dict()

    pydantic.v1.json.ENCODERS_BY_TYPE[Timeout] = encode_openai_timeout


def obj_id_of_obj(obj: dict, prefix="obj"):
    """
    Create an id from a json-able structure/definition. Should produce the same
    name if definition stays the same.
    """

    return f"{prefix}_hash_{mj.hash(obj)}"


def json_str_of_obj(
    obj: Any, *args, redact_keys: bool = False, **kwargs
) -> str:
    """
    Encode the given json object as a string.
    """

    return json.dumps(
        jsonify(obj, *args, redact_keys=redact_keys, **kwargs),
        default=json_default
    )


def json_default(obj: Any) -> str:
    """
    Produce a representation of an object which does not have a json serializer.
    """

    # Try the encoders included with pydantic first (should handle things like
    # Datetime, and our additional encoders above):
    try:
        return pydantic.v1.json.pydantic_encoder(obj)

    except:
        # Otherwise give up and indicate a non-serialization.
        return noserio(obj)


ALL_SPECIAL_KEYS = set([CIRCLE, ERROR, CLASS_INFO, NOSERIO])


def jsonify_for_ui(*args, **kwargs):
    """Options for jsonify common to UI displays.
    
    Redacts keys and hides special fields introduced by trulens.
    """

    return jsonify(*args, **kwargs, redact_keys=True, skip_specials=True)


def jsonify(
    obj: Any,
    dicted: Optional[Dict[int, JSON]] = None,
    instrument: Optional['Instrument'] = None,
    skip_specials: bool = False,
    redact_keys: bool = False,
    include_excluded: bool = True,
    depth: int = 0,
    max_depth: int = 256
) -> JSON:
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
    # they are stateful if iterated. That is, they might be iteratable only once
    # and iterating will break their user's interfaces.
    """
    if isinstance(obj, typing.Iterator):
        raise ValueError("Cannot jsonify an iterator object.")
    if inspect.isawaitable(obj):
        raise ValueError("Cannot jsonify an awaitable object.")
    if inspect.isgenerator(obj):
        raise ValueError("Cannot jsonify a generator object.")
    if inspect.iscoroutine(obj):
        raise ValueError("Cannot jsonify a coroutine object.")
    if inspect.isasyncgen(obj):
        raise ValueError("Cannot jsonify an async generator object.")
    if inspect.isasyncgenfunction(obj):
        raise ValueError("Cannot jsonify an async generator function.")
    if inspect.iscoroutinefunction(obj):
        raise ValueError("Cannot jsonify a coroutine function.")
    if inspect.isgeneratorfunction(obj):
        raise ValueError("Cannot jsonify a generator function.")
    """

    if depth > max_depth:
        logger.debug(
            "Max depth reached for jsonify of object type '%s'.",
            type(obj)
        ) # careful about str(obj) in case it is recursive infinitely.

        return noserio(obj)

    skip_excluded = not include_excluded
    # Hack so that our models do not get exludes dumped which causes many
    # problems. Another variable set here so we can recurse with the original
    # include_excluded .
    if isinstance(obj, SerialModel):
        skip_excluded = True

    from trulens_eval.instruments import Instrument

    if instrument is None:
        instrument = Instrument()

    dicted = dicted or {}

    if skip_specials:

        def recur_key(k):
            return isinstance(k, JSON_BASES) and k not in ALL_SPECIAL_KEYS
    else:

        def recur_key(k):
            return isinstance(k, JSON_BASES)

    if id(obj) in dicted:
        if skip_specials:
            return None

        return {CIRCLE: id(obj)}

    if isinstance(obj, JSON_BASES):
        if redact_keys and isinstance(obj, str):
            return redact_value(obj)

        return obj

    # TODO: remove eventually
    if isinstance(obj, SerialBytes):
        return obj.model_dump()

    if isinstance(obj, Path):
        return str(obj)

    if type(obj) in pydantic.v1.json.ENCODERS_BY_TYPE:
        return pydantic.v1.json.ENCODERS_BY_TYPE[type(obj)](obj)

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
            max_depth=max_depth
        )

    content = None

    if isinstance(obj, Enum):
        content = obj.name

    elif isinstance(obj, Dict):
        forward_value = {}
        new_dicted[id(obj)] = forward_value
        forward_value.update({k: recur(v) for k, v in obj.items() if recur_key(k)})

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in forward_value.items():
                forward_value[k] = redact_value(v=v, k=k)

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

    elif isinstance(obj, pydantic.BaseModel):
        # Not even trying to use pydantic.dict here.

        if isinstance(obj, Lens):  # special handling of paths
            return obj.model_dump()

        forward_value = {}
        new_dicted[id(obj)] = forward_value
        forward_value.update(
            {
                k: recur(safe_getattr(obj, k))
                for k, v in obj.model_fields.items()
                if (not skip_excluded or not v.exclude) and recur_key(k)
            }
        )

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in forward_value.items():
                forward_value[k] = redact_value(v=v, k=k)

        content = forward_value

    elif isinstance(obj, pydantic.v1.BaseModel):
        # TODO: DEDUP with pydantic.BaseModel case

        # Not even trying to use pydantic.dict here.

        forward_value = {}
        new_dicted[id(obj)] = forward_value
        forward_value.update(
            {
                k: recur(safe_getattr(obj, k))
                for k, v in obj.__fields__.items()
                if (not skip_excluded or not v.field_info.exclude) and
                recur_key(k)
            }
        )

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in forward_value.items():
                forward_value[k] = redact_value(v=v, k=k)

        content = forward_value

    elif dataclasses.is_dataclass(type(obj)):
        # NOTE: cannot use dataclasses.asdict as that may fail due to its use of
        # copy.deepcopy.

        forward_value = {}
        new_dicted[id(obj)] = forward_value

        forward_value.update(
            {
                f.name: recur(safe_getattr(obj, f.name))
                for f in dataclasses.fields(obj)
                if recur_key(f.name)
            }
        )

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in forward_value.items():
                forward_value[k] = redact_value(v=v, k=k)

        content = forward_value

    elif instrument.to_instrument_object(obj):

        forward_value = {}
        new_dicted[id(obj)] = forward_value

        kvs = clean_attributes(obj, include_props=True)

        # TODO(piotrm): object walks redo
        forward_value.update(
            {
                k: recur(v) for k, v in kvs.items() if recur_key(k) and (
                    isinstance(v, JSON_BASES) or isinstance(v, Dict) or
                    isinstance(v, Sequence) or
                    instrument.to_instrument_object(v)
                )
            }
        )

        content = forward_value

    else:
        logger.debug(
            "Do not know how to jsonify an object of type '%s'.",
            type(obj)
        ) # careful about str(obj) in case it is recursive infinitely.

        content = noserio(obj)

    # Add class information for objects that are to be instrumented, known as
    # "components".
    if not skip_specials and isinstance(content, dict) and not isinstance(
            obj, dict) and (instrument.to_instrument_object(obj) or
                            isinstance(obj, WithClassInfo)):

        content[CLASS_INFO] = Class.of_class(
            cls=obj.__class__, with_bases=True
        ).model_dump()

    if not isinstance(obj, Lens) and safe_hasattr(obj, "jsonify_extra"):
        # Problem with Lens and similar objects: they always say they have every attribute.

        content = obj.jsonify_extra(content)

    return content
