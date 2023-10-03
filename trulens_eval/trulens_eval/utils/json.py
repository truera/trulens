"""
Json utilities and serialization utilities dealing with json.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
import json
import logging
from pathlib import Path
from pprint import PrettyPrinter
from typing import (
    Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple, TypeVar,
    Union
)

from merkle_json import MerkleJson
import pydantic

from trulens_eval.keys import redact_value
from trulens_eval.utils.pyschema import WithClassInfo, _clean_attributes
from trulens_eval.utils.pyschema import _safe_getattr
from trulens_eval.utils.pyschema import CIRCLE
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import CLASS_INFO
from trulens_eval.utils.pyschema import ERROR
from trulens_eval.utils.pyschema import NOSERIO
from trulens_eval.utils.pyschema import noserio
from trulens_eval.utils.serial import JSON, SerialBytes
from trulens_eval.utils.serial import JSON_BASES
from trulens_eval.utils.serial import JSONPath

logger = logging.getLogger(__name__)
pp = PrettyPrinter()

T = TypeVar("T")

mj = MerkleJson()


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
    Produce a representation of an object which cannot be json-serialized.
    """

    # Try the encoders included with pydantic first (should handle things like
    # Datetime):
    try:
        return pydantic.json.pydantic_encoder(obj)
    except:
        # Otherwise give up and indicate a non-serialization.
        return noserio(obj)


ALL_SPECIAL_KEYS = set([CIRCLE, ERROR, CLASS_INFO, NOSERIO])

def jsonify_for_ui(*args, **kwargs):
    """
    Options for jsonify common to UI displays. Redact keys and hide special
    fields.
    """
    return jsonify(*args, **kwargs, redact_keys=True, skip_specials=True)


# TODO: refactor to somewhere else or change instrument to a generic filter
def jsonify(
    obj: Any,
    dicted: Optional[Dict[int, JSON]] = None,
    instrument: Optional['Instrument'] = None,
    skip_specials: bool = False,
    redact_keys: bool = False
) -> JSON:
    """
    Convert the given object into types that can be serialized in json.

    Args:

        - obj: Any -- the object to jsonify.

        - dicted: Optional[Dict[int, JSON]] -- the mapping from addresses of
          already jsonifed objects (via id) to their json.

        - instrument: Optional[Instrument] -- instrumentation functions for
          checking whether to recur into components of `obj`.

        - skip_specials: bool (default is False) -- if set, will remove
          specially keyed structures from the json. These have keys that start
          with "__tru_".

        - redact_keys: bool (default is False) -- if set, will redact secrets
          from the output. Secrets are detremined by `keys.py:redact_value` .

    Returns:

        JSON | Sequence[JSON]
    """

    from trulens_eval.instruments import Instrument

    instrument = instrument or Instrument()
    dicted = dicted or dict()

    if skip_specials:
        recur_key = lambda k: k not in ALL_SPECIAL_KEYS
    else:
        recur_key = lambda k: True

    if id(obj) in dicted:
        if skip_specials:
            return None
        else:
            return {CIRCLE: id(obj)}

    if isinstance(obj, JSON_BASES):
        if redact_keys and isinstance(obj, str):
            return redact_value(obj)
        else:
            return obj

    # TODO: remove eventually
    if isinstance(obj, SerialBytes):
        return obj.dict()

    if isinstance(obj, Path):
        return str(obj)

    if type(obj) in pydantic.json.ENCODERS_BY_TYPE:
        return obj

    # TODO: should we include duplicates? If so, dicted needs to be adjusted.
    new_dicted = {k: v for k, v in dicted.items()}

    recur = lambda o: jsonify(
        obj=o,
        dicted=new_dicted,
        instrument=instrument,
        skip_specials=skip_specials,
        redact_keys=redact_keys
    )

    content = None

    if isinstance(obj, Enum):
        content = obj.name

    elif isinstance(obj, Dict):
        temp = {}
        new_dicted[id(obj)] = temp
        temp.update({k: recur(v) for k, v in obj.items() if recur_key(k)})

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in temp.items():
                temp[k] = redact_value(v=v, k=k)

        content = temp

    elif isinstance(obj, Sequence):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (recur(v) for v in obj):
            temp.append(x)

        content = temp

    elif isinstance(obj, Set):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (recur(v) for v in obj):
            temp.append(x)

        content = temp

    elif isinstance(obj, pydantic.BaseModel):
        # Not even trying to use pydantic.dict here.

        temp = {}
        new_dicted[id(obj)] = temp
        temp.update(
            {
                k: recur(_safe_getattr(obj, k))
                for k, v in obj.__fields__.items()
                if not v.field_info.exclude and recur_key(k)
            }
        )

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in temp.items():
                temp[k] = redact_value(v=v, k=k)

        content = temp

    elif dataclasses.is_dataclass(type(obj)):
        # NOTE: cannot use dataclasses.asdict as that may fail due to its use of
        # copy.deepcopy.

        temp = {}
        new_dicted[id(obj)] = temp

        temp.update(
            {
                f.name: recur(_safe_getattr(obj, f.name))
                for f in dataclasses.fields(obj)
                if recur_key(f.name)
            }
        )

        # Redact possible secrets based on key name and value.
        if redact_keys:
            for k, v in temp.items():
                temp[k] = redact_value(v=v, k=k)

        content = temp

    elif instrument.to_instrument_object(obj):

        temp = {}
        new_dicted[id(obj)] = temp

        kvs = _clean_attributes(obj)

        temp.update(
            {
                k: recur(v) for k, v in kvs.items() if recur_key(k) and (
                    isinstance(v, JSON_BASES) or isinstance(v, Dict) or
                    isinstance(v, Sequence) or
                    instrument.to_instrument_object(v)
                )
            }
        )

        content = temp

    else:
        logger.debug(
            f"Do not know how to jsonify an object '{str(obj)[0:32]}' of type '{type(obj)}'."
        )

        content = noserio(obj)

    # Add class information for objects that are to be instrumented, known as
    # "components".
    if instrument.to_instrument_object(obj) or isinstance(obj, WithClassInfo):
        content[CLASS_INFO] = Class.of_class(
            cls=obj.__class__, with_bases=True
        ).dict()

    if not isinstance(obj, JSONPath) and hasattr(obj, "jsonify_extra"):
        # Problem with JSONPath and similar objects: they always say they have every attribute.

        content = obj.jsonify_extra(content)

    return content
