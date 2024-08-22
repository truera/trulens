from __future__ import annotations

import logging
from typing import (
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAliasType,
    TypeVar,
    Union,
)

from opentelemetry.util import types as ot_types
from trulens.core.utils import python as python_utils
from trulens.core.utils.serial import Lens

logger = logging.getLogger(__name__)

# Type alises

A = TypeVar("A")
B = TypeVar("B")

TTimestamp = int
"""Type of timestamps in spans.
64 bit int representing nanoseconds since epoch as per OpenTelemetry.
"""
NUM_TIMESTAMP_BITS = 64

TSpanID = int
"""Type of span identifiers.
64 bit int as per OpenTelemetry.
"""
NUM_SPANID_BITS = 64
"""Number of bits in a span identifier."""

TTraceID = int
"""Type of trace identifiers.
128 bit int as per OpenTelemetry.
"""
NUM_TRACEID_BITS = 128
"""Number of bits in a trace identifier."""

TLensedBaseType = Union[str, int, float, bool, python_utils.NoneType]
"""Type of base types in span attributes.
!!! Warning
    OpenTelemetry does not allow None as an attribute value. However, we allow
    it in lensed attributes.
"""

TLensedAttributeValue = TypeAliasType(
    "TLensedAttributeValue",
    Union[
        str,
        int,
        float,
        bool,
        python_utils.NoneType,  # NOTE(piotrm): None is not technically allowed as an attribute value.
        Sequence["TLensedAttributeValue"],
        "TLensedAttributes",
    ],
)
"""Type of values in span attributes."""

# NOTE(piotrm): pydantic will fail if you specify a recursive type alias without
# the TypeAliasType schema as above.

TLensedAttributes = Dict[str, TLensedAttributeValue]


def flatten_value(
    v: TLensedAttributeValue, path: Optional[Lens] = None
) -> Iterable[Tuple[Lens, ot_types.AttributeValue]]:
    """Flatten a lensed value into OpenTelemetry attribute values."""

    if path is None:
        path = Lens()

    # if v is None:
    # OpenTelemetry does not allow None as an attribute value. Unsure what
    # is best to do here. Returning "None" for now.
    #    yield (path, "None")

    elif isinstance(v, TLensedBaseType):
        yield (path, v)

    elif isinstance(v, Sequence) and all(
        isinstance(e, TLensedBaseType) for e in v
    ):
        yield (path, v)

    elif isinstance(v, Sequence):
        for i, e in enumerate(v):
            yield from flatten_value(v=e, path=path[i])

    elif isinstance(v, Mapping):
        for k, e in v.items():
            yield from flatten_value(v=e, path=path[k])

    else:
        raise ValueError(
            f"Do not know how to flatten value of type {type(v)} to OTEL attributes."
        )


def flatten_lensed_attributes(
    m: TLensedAttributes, path: Optional[Lens] = None, prefix: str = ""
) -> ot_types.Attributes:
    """Flatten lensed attributes into OpenTelemetry attributes."""

    if path is None:
        path = Lens()

    ret = {}
    for k, v in m.items():
        if k.startswith(prefix):
            # Only flattening those attributes that begin with `prefix` are
            # those are the ones coming from trulens_eval.
            for p, a in flatten_value(v, path[k]):
                ret[str(p)] = a
        else:
            ret[k] = v

    return ret
