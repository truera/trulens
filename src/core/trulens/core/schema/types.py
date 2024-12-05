"""
Type aliases and basic types that share interpretation across different systems:
python, OTEL, and SQL.

This file is here to consolidate the many places where these types appear or
where aliases to them ended up.

Do not make this module depend on any in TruLens except things which themselves
have little to no internal dependencies.
"""

# TODO: move other type aliases for basic types to here:
# - trulens.core.database.orm:py has some
# -

from __future__ import annotations

import datetime
from enum import Enum
import random
import sys
import time
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import uuid

from opentelemetry import trace as trace_api
from opentelemetry.trace import span as span_api
from opentelemetry.trace import status as status_api
from opentelemetry.util import types as types_api
from sqlalchemy import BINARY
from sqlalchemy import JSON
from sqlalchemy import SMALLINT
from sqlalchemy import TIMESTAMP
from sqlalchemy import VARCHAR
from trulens.core._utils.pycompat import NoneType  # import style exception
from trulens.core._utils.pycompat import Type  # import style exception
from trulens.core._utils.pycompat import TypeAlias  # import style exception
from trulens.core._utils.pycompat import TypeAliasType  # import style exception
from trulens.core.utils import serial as serial_utils
from trulens.otel.semconv import trace as truconv

RecordID: TypeAlias = str
"""Unique identifier for a record.

By default these hashes of record content as json.
[Record.record_id][trulens.core.schema.record.Record.record_id].
"""

CallID: TypeAlias = str
"""Unique identifier for a record app call.

See [RecordAppCall.call_id][trulens.core.schema.record.RecordAppCall.call_id].
"""


def new_call_id() -> CallID:
    """Generate a new call id."""
    return str(uuid.uuid4())


AppID: TypeAlias = str
"""Unique identifier for an app.

By default these are hashes of app content as json.
See [AppDefinition.app_id][trulens.core.schema.app.AppDefinition.app_id].
"""

AppName: TypeAlias = str
"""Unique App name.

See [AppDefinition.app_name][trulens.core.schema.app.AppDefinition.app_name].
"""

AppVersion: TypeAlias = str
"""Version identifier for an app.

See [AppDefinition.app_version][trulens.core.schema.app.AppDefinition.app_version].
"""

Tags: TypeAlias = str
"""Tags for an app or record.

See [AppDefinition.tags][trulens.core.schema.app.AppDefinition.tags] and
[Record.tags][trulens.core.schema.record.Record.tags].
"""

Metadata: TypeAlias = Dict
"""Metadata for an app, record, groundtruth, or dataset.

See [AppDefinition.metadata][trulens.core.schema.app.AppDefinition.metadata],
[Record.meta][trulens.core.schema.record.Record.meta], [GroundTruth.meta][trulens.core.schema.groundtruth.GroundTruth.meta], and
[Dataset.meta][trulens.core.schema.dataset.Dataset.meta].
"""

FeedbackDefinitionID: TypeAlias = str
"""Unique identifier for a feedback definition.

By default these are hashes of feedback definition content as json. See
[FeedbackDefinition.feedback_definition_id][trulens.core.schema.feedback.FeedbackDefinition.feedback_definition_id].
"""

FeedbackResultID: TypeAlias = str
"""Unique identifier for a feedback result.

By default these are hashes of feedback result content as json. See
[FeedbackResult.feedback_result_id][trulens.core.schema.feedback.FeedbackResult].
"""

GroundTruthID: TypeAlias = str
"""Unique identifier for a groundtruth.

By default these are hashes of ground truth content as json.
"""

DatasetID: TypeAlias = str
"""Unique identifier for a dataset.

By default these are hashes of dataset content as json.
"""

# The rest is a type organization introduced with EXPERIMENTAL(otel_tracing).

T = TypeVar("T")

O = TypeVar("O")  # noqa: E741
"""Types for values in OTEL representations of spans or otherwise."""

P = TypeVar("P")
"""Types for values in python."""

S = TypeVar("S")
"""Types for values provided to or retrieved from sqlalchemy."""


class TypeInfo(Generic[O, P, S]):
    """Container for types and conversions between types used to represent
    values that need to be stored/represented in different places.

    The places are at least python and sql and optionally in OTEL.

    The capabilities this class exposes are:

    - Type aliases in the form of `{PY,SQL,OTEL}_TYPE`.

    - Type union for the above.

    -
    """

    OTEL_TYPE: Optional[Type[O]] = None
    """Type for OTEL values.

    None if type info is not relevant to OTEL.
    """

    NUM_BITS: Optional[int] = None
    """Number of bits in values of this type.

    Note that when represented in python, OTEL, or SQL, the number of bits may
    vary but may be specified by a spec, like OTEL, to be a certain number of
    bits regardless."""

    PY_TYPE: Type[P]
    """Type for python values."""

    SQL_TYPE: Type[S]
    """Type for values understood by sqlalchemy to be representing in the
    database as the column type `SQL_SCHEMA_TYPE`."""

    SQL_SCHEMA_TYPE: Type
    """SQL column type for the column type."""

    UNION_TYPE: Union[Type[O], Type[P], Type[S]]
    """Union of all types that can be used to represent values of this type
    except the schema type."""

    TYPES: Tuple[Type[O], Type[P], Type[S]]
    """Tuple of the above so that isinstance can be used."""

    @classmethod
    def py(cls, val: TypeInfo.UNION_TYPE) -> P:
        """Convert a value to a python value."""
        if isinstance(val, cls.PY_TYPE):
            return val
        if isinstance(val, cls.SQL_TYPE):
            return cls.py_of_sql(val)
        if cls.OTEL_TYPE is not None and isinstance(val, cls.OTEL_TYPE):
            return cls.py_of_otel(val)

        raise TypeError(f"Cannot convert value of type {type(val)} to python.")

    @classmethod
    def otel(cls, val: TypeInfo.UNION_TYPE) -> O:
        """Convert a value to the otel representation."""

        cls._assert_has_otel()

        if isinstance(val, cls.OTEL_TYPE):
            return val
        if isinstance(val, cls.PY_TYPE):
            return cls.otel_of_py(val)
        if isinstance(val, cls.SQL_TYPE):
            return cls.otel_of_sql(val)

        raise TypeError(f"Cannot convert value of type {type(val)} to otel.")

    @classmethod
    def sql(cls, val: TypeInfo.UNION_TYPE) -> S:
        """Convert a value to the sql representation."""

        if isinstance(val, cls.SQL_TYPE):
            return val
        if isinstance(val, cls.PY_TYPE):
            return cls.sql_of_py(val)
        if cls.OTEL_TYPE is not None and isinstance(val, cls.OTEL_TYPE):
            return cls.sql_of_otel(val)

        raise TypeError(f"Cannot convert value of type {type(val)} to sql.")

    @classmethod
    def default_py(cls) -> P:
        """Default python value for this type."""
        return cls.rand_py()

    @classmethod
    def default_sql(cls) -> S:
        """Default sql value for this type."""
        return cls.rand_sql()

    @classmethod
    def rand_py(cls) -> P:
        """Generate a new random python value of this type."""
        if cls.rand_otel is not TypeInfo.rand_otel:
            return cls.py_of_otel(cls.rand_otel())
        if cls.rand_sql is not TypeInfo.rand_sql:
            return cls.py_of_sql(cls.rand_sql())
        raise NotImplementedError("Python type generation not implemented.")

    @classmethod
    def rand_sql(cls) -> S:
        """Generate a new random sql value of this type."""
        if cls.rand_otel is not TypeInfo.rand_otel:
            return cls.sql_of_otel(cls.rand_otel())
        if cls.rand_py is not TypeInfo.rand_py:
            return cls.sql_of_py(cls.rand_py())

        raise NotImplementedError("SQL type generation not implemented.")

    @classmethod
    def sql_of_py(cls, py_value: P) -> S:
        """Convert a python value to a sql value."""

        if cls.PY_TYPE is cls.SQL_TYPE:
            return py_value

        if (
            cls.sql_of_otel is not TypeInfo.sql_of_otel
            and cls.otel_of_py is not TypeInfo.otel_of_py
        ):
            return cls.sql_of_otel(cls.otel_of_py(py_value))

        raise NotImplementedError

    @classmethod
    def py_of_sql(cls, sql_value: S) -> P:
        """Convert a sql value to a python value."""

        if cls.PY_TYPE is cls.SQL_TYPE:
            return sql_value

        if (
            cls.py_of_otel is not TypeInfo.py_of_otel
            and cls.otel_of_sql is not TypeInfo.otel_of_sql
        ):
            return cls.py_of_otel(cls.otel_of_sql(sql_value))

        raise NotImplementedError

    @classmethod
    def _assert_has_otel(cls) -> None:
        if cls.OTEL_TYPE is None:
            raise NotImplementedError(
                f"{cls.__name__} does not support OTEL values."
            )

    @classmethod
    def rand_otel(cls) -> O:
        """Generate a new random otel value of this type."""

        cls._assert_has_otel()

        if cls.rand_py is not TypeInfo.rand_py:
            return cls.otel_of_py(cls.rand_py())
        if cls.rand_sql is not TypeInfo.rand_sql:
            return cls.otel_of_sql(cls.rand_sql())

        raise NotImplementedError("OTEL type generation not implemented.")

    @classmethod
    def default_otel(cls) -> O:
        """Default otel value for this type."""

        cls._assert_has_otel()

        return cls.rand_otel()

    @classmethod
    def otel_of_py(cls, py_value: P) -> O:
        """Convert a python value to an otel value."""

        cls._assert_has_otel()

        if cls.OTEL_TYPE is cls.PY_TYPE:
            return py_value

        if (
            cls.otel_of_sql is not TypeInfo.otel_of_sql
            and cls.sql_of_py is not TypeInfo.sql_of_py
        ):
            return cls.otel_of_sql(cls.sql_of_py(py_value))

        raise NotImplementedError

    @classmethod
    def py_of_otel(cls, otel_value: O) -> P:
        """Convert an otel value to a python value."""

        cls._assert_has_otel()

        if cls.PY_TYPE is cls.OTEL_TYPE:
            return otel_value

        if (
            cls.py_of_sql is not TypeInfo.py_of_sql
            and cls.sql_of_otel is not TypeInfo.sql_of_otel
        ):
            return cls.py_of_sql(cls.sql_of_otel(otel_value))

        raise NotImplementedError

    @classmethod
    def otel_of_sql(cls, sql_value: S) -> O:
        """Convert a sql value to an otel value."""

        cls._assert_has_otel()

        if cls.OTEL_TYPE is cls.SQL_TYPE:
            return sql_value

        if (
            cls.otel_of_py is not TypeInfo.otel_of_py
            and cls.py_of_sql is not TypeInfo.py_of_sql
        ):
            return cls.otel_of_py(cls.py_of_sql(sql_value))

        raise NotImplementedError

    @classmethod
    def sql_of_otel(cls, otel_value: O) -> S:
        """Convert an otel value to a sql value."""

        cls._assert_has_otel()

        if cls.SQL_TYPE is cls.OTEL_TYPE:
            return otel_value

        if (
            cls.sql_of_py is not TypeInfo.sql_of_py
            and cls.py_of_otel is not TypeInfo.py_of_otel
        ):
            return cls.sql_of_py(cls.py_of_otel(otel_value))

        raise NotImplementedError


class SpanID(TypeInfo[int, int, bytes]):
    """Span ID type.

    This type is for supporting OTEL hence its requirements come from there. In
    OTEL and python it is a 64-bit integer. In the database, it is a binary
    column with 64 bits or 8 bytes and read as bytes.
    """

    NUM_BITS: int = 64
    """Number of bits in a span identifier."""

    OTEL_TYPE: Type = int
    PY_TYPE: Type = int
    SQL_TYPE: Type = bytes
    SQL_SCHEMA_TYPE: Type = BINARY(NUM_BITS // 8)

    UNION_TYPE: Type = Union[int, bytes]
    TYPES = (int, bytes)

    INVALID_OTEL = span_api.INVALID_SPAN_ID
    """Span ID for non-recording or invalid spans."""

    @classmethod
    def rand_otel(cls) -> int:
        return int(
            random.getrandbits(cls.NUM_BITS) & span_api._SPAN_ID_MAX_VALUE
        )

    @classmethod
    def rand_py(cls) -> int:
        return cls.rand_otel()

    @classmethod
    def rand_sql(cls) -> bytes:
        return cls.sql_of_py(cls.rand_py())

    @classmethod
    def otel_of_py(cls, py_value: int) -> int:
        return py_value

    @classmethod
    def py_of_otel(cls, otel_value: int) -> int:
        return otel_value

    @classmethod
    def sql_of_py(cls, py_value: int) -> bytes:
        return py_value.to_bytes(cls.NUM_BITS // 8, byteorder="big")

    @classmethod
    def py_of_sql(cls, sql_value: bytes) -> int:
        return int.from_bytes(sql_value, byteorder="big")


class TraceID(TypeInfo[int, int, bytes]):
    """Trace ID type.

    This type is for supporting OTEL hence its requirements come from there. In
    OTEL and python it is a 128-bit integer. In the database, it is a binary
    column with 128 bits or 16 bytes.
    """

    NUM_BITS: int = 64
    """Number of bits in a span identifier."""

    OTEL_TYPE: Type = int
    PY_TYPE: Type = int
    SQL_TYPE: Type = bytes
    SQL_SCHEMA_TYPE: Type = BINARY(NUM_BITS // 8)

    UNION_TYPE: Type = Union[int, bytes]
    TYPES: Tuple = (int, bytes)

    INVALID_OTEL = span_api.INVALID_TRACE_ID
    """Trace ID for non-recording or invalid spans."""

    @classmethod
    def rand_otel(cls) -> int:
        return int(
            random.getrandbits(cls.NUM_BITS)
            & trace_api.span._TRACE_ID_MAX_VALUE
        )

    @classmethod
    def rand_py(cls) -> int:
        return cls.rand_otel()

    @classmethod
    def rand_sql(cls) -> bytes:
        return cls.sql_of_py(cls.rand_py())

    @classmethod
    def otel_of_py(cls, py_value: int) -> int:
        return py_value

    @classmethod
    def py_of_otel(cls, otel_value: int) -> int:
        return otel_value

    @classmethod
    def sql_of_py(cls, py_value: int) -> bytes:
        return py_value.to_bytes(cls.NUM_BITS // 8, byteorder="big")

    @classmethod
    def py_of_sql(cls, sql_value: bytes) -> int:
        return int.from_bytes(sql_value, byteorder="big")


class StrAsVarChar(TypeInfo[str, str, str]):
    """Types that are strings in python,otel,sql interface and VARCHAR in SQL column."""

    NUM_BYTES: int = 256

    OTEL_TYPE: Type = str
    PY_TYPE: Type = str
    SQL_TYPE: Type = str
    SQL_SCHEMA_TYPE: Type = VARCHAR(NUM_BYTES)

    UNION_TYPE: Type = str
    TYPES: Tuple = (str,)

    @classmethod
    def rand_otel(cls) -> str:
        return str(uuid.uuid4())

    @classmethod
    def rand_py(cls) -> str:
        return cls.rand_otel()

    @classmethod
    def rand_sql(cls) -> str:
        return cls.rand_otel()

    @classmethod
    def otel_of_py(cls, py_value: str) -> str:
        return py_value

    @classmethod
    def py_of_otel(cls, otel_value: str) -> str:
        return otel_value

    @classmethod
    def sql_of_py(cls, py_value: str) -> str:
        return py_value

    @classmethod
    def py_of_sql(cls, sql_value: str) -> str:
        return sql_value


class TraceRecordID(StrAsVarChar):
    """Types for representing record ids in traces/spans."""

    NUM_BYTES: int = 32


class ListAsJSON(TypeInfo[List[O], List[P], List[S]], Generic[O, P, S]):
    """Lists stored as JSON in the database."""

    ETI: TypeInfo[O, P, S]
    """TypeInfo for elements."""

    OTEL_TYPE: Type = List[O]
    PY_TYPE: Type = List[P]
    SQL_TYPE: Type = List[S]
    SQL_SCHEMA_TYPE: Type = JSON

    @classmethod
    def otel_of_py(cls, py_value: List[P]) -> List[O]:
        return [cls.ETI.otel_of_py(val) for val in py_value]

    @classmethod
    def py_of_otel(cls, otel_value: List[O]) -> List[P]:
        return [cls.ETI.py_of_otel(val) for val in otel_value]

    @classmethod
    def sql_of_py(cls, py_value: List[P]) -> List[S]:
        return [cls.ETI.sql_of_py(val) for val in py_value]

    @classmethod
    def py_of_sql(cls, sql_value: List[S]) -> List[P]:
        return [cls.ETI.py_of_sql(val) for val in sql_value]


class DictAsJSON(
    TypeInfo[Dict[str, O], Dict[str, P], Dict[str, S]], Generic[O, P, S]
):
    """Dicts of str keys stored as JSON in the database."""

    ETI: TypeInfo[O, P, S]
    """TypeInfo for elements."""

    OTEL_TYPE: Type = Dict[str, O]
    PY_TYPE: Type = Dict[str, P]
    SQL_TYPE: Type = Dict[str, S]
    SQL_SCHEMA_TYPE: Type = JSON

    @classmethod
    def otel_of_py(cls, py_value: Dict[str, P]) -> Dict[str, O]:
        return {k: cls.ETI.otel_of_py(val) for k, val in py_value.items()}

    @classmethod
    def py_of_otel(cls, otel_value: Dict[str, O]) -> Dict[str, P]:
        return {k: cls.ETI.py_of_otel(val) for k, val in otel_value.items()}

    @classmethod
    def sql_of_py(cls, py_value: Dict[str, P]) -> Dict[str, S]:
        return {k: cls.ETI.sql_of_py(val) for k, val in py_value.items()}

    @classmethod
    def py_of_sql(cls, sql_value: Dict[str, S]) -> Dict[str, P]:
        return {k: cls.ETI.py_of_sql(val) for k, val in sql_value.items()}


class TraceRecordIDs(
    DictAsJSON[
        TraceRecordID.OTEL_TYPE, TraceRecordID.PY_TYPE, TraceRecordID.SQL_TYPE
    ]
):
    """Type for representing multiple trace record ids.

    This is a list of trace record ids. It is a list of `TraceRecordID`.
    """

    ETI = TraceRecordID


class SpanName(StrAsVarChar):
    """Span names."""

    NUM_BYTES = 32
    # TODO: get from otel spec


E = TypeVar("E", bound=Enum)
"""Enum types."""


class IntEnumAsSmallInt(TypeInfo[E, E, int], Generic[E]):
    """Enum types that are stored as integers in the database."""

    OTEL_TYPE: Type[E]  # to fill in by subclass
    PY_TYPE: Type[E]  # to fill in by subclass
    SQL_TYPE: Type = int
    SQL_SCHEMA_TYPE: Type = (
        SMALLINT  # override in subclass if bigger int needed
    )

    UNION_TYPE: Type  # to fill in by subclass
    TYPES: Tuple  # to fill in by subclass

    @classmethod
    def sql_of_py(cls, py_value: E) -> int:
        return py_value.value

    @classmethod
    def py_of_sql(cls, sql_value: int) -> E:
        return cls.PY_TYPE(sql_value)


class StrEnumAsVarChar(TypeInfo[E, E, str], Generic[E]):
    """Enum types that are stored as varchar in the database."""

    OTEL_TYPE: Type[E]  # to fill in by subclass
    PY_TYPE: Type[E]  # to fill in by subclass
    SQL_TYPE: Type = str
    SQL_SCHEMA_TYPE: Type = VARCHAR(
        16
    )  # override in subclass if bigger int needed

    UNION_TYPE: Type  # to fill in by subclass
    TYPES: Tuple  # to fill in by subclass

    @classmethod
    def sql_of_py(cls, py_value: E) -> str:
        return py_value.value

    @classmethod
    def py_of_sql(cls, sql_value: str) -> E:
        return cls.PY_TYPE(sql_value)


class SpanType(StrEnumAsVarChar):
    """Span type enum."""

    OTEL_TYPE = truconv.SpanAttributes.SpanType
    PY_TYPE = truconv.SpanAttributes.SpanType
    UNION_TYPE = Union[truconv.SpanAttributes.SpanType, str]
    TYPES = (truconv.SpanAttributes.SpanType, str)


class SpanTypes(
    ListAsJSON[SpanType.OTEL_TYPE, SpanType.PY_TYPE, SpanType.SQL_TYPE]
):
    """Type for representing multiple span types.

    This is a list of span types. It is a list of `SpanType`.
    """

    ETI = SpanType


class SpanStatusCode(IntEnumAsSmallInt):
    """Span status enum."""

    OTEL_TYPE = status_api.StatusCode
    PY_TYPE = status_api.StatusCode
    UNION_TYPE = Union[status_api.StatusCode, int]
    TYPES = (status_api.StatusCode, int)


class StatusDescription(StrAsVarChar):
    NUM_BYTES = 1024
    # TODO: get from otel spec


class SpanKind(IntEnumAsSmallInt):
    """Span kind enum."""

    OTEL_TYPE = trace_api.SpanKind
    PY_TYPE = trace_api.SpanKind
    UNION_TYPE = Union[trace_api.SpanKind, int]
    TYPES = (trace_api.SpanKind, int)


class Timestamp(TypeInfo[int, datetime.datetime, datetime.datetime]):
    """Timestamp type.

    This type is for supporting OTEL hence its requirements come from there. In
    OTEL it is a 64-bit integer representing the number of nano seconds since
    the epoch. In python, it is a [datetime][datetime.datetime]. In the
    database, it is the TIMESTAMP sql column type.

    Default values are "now" and random is not supported.
    """

    NUM_BITS = 64
    """Number of bits in a span identifier."""

    OTEL_TYPE: Type = int
    PY_TYPE: Type = datetime.datetime
    SQL_TYPE: Type = datetime.datetime
    SQL_SCHEMA_TYPE: Type = TIMESTAMP

    UNION_TYPE: Type = Union[int, datetime.datetime]

    TYPES: Tuple = (int, datetime.datetime)

    @classmethod
    def default_py(cls) -> datetime.datetime:
        return datetime.datetime.now()

    @classmethod
    def default_sql(cls) -> datetime.datetime:
        return cls.default_py()

    @classmethod
    def default_otel(cls) -> int:
        return time.time_ns()

    @classmethod
    def rand_otel(cls) -> int:
        raise NotImplementedError("Timestamps are not meant to be random.")

    @classmethod
    def rand_py(cls) -> datetime.datetime:
        raise NotImplementedError("Timestamps are not meant to be random.")

    @classmethod
    def rand_sql(cls) -> datetime.datetime:
        raise NotImplementedError("Timestamps are not meant to be random.")

    @classmethod
    def otel_of_py(cls, py_value: datetime.datetime) -> int:
        return int(py_value.timestamp() * 1e9)

    @classmethod
    def py_of_otel(cls, otel_value: int) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(otel_value / 1e9)

    @classmethod
    def sql_of_py(cls, py_value: datetime.datetime) -> datetime.datetime:
        return py_value

    @classmethod
    def py_of_sql(cls, sql_value: datetime.datetime) -> datetime.datetime:
        return sql_value


# OTEL Attributes-related


def lens_of_flat_key(key: str) -> serial_utils.Lens:
    """Convert a flat dict key to a lens."""
    lens = serial_utils.Lens()
    for step in key.split("."):
        lens = lens[step]

    return lens


if sys.version_info >= (3, 9):
    TLensedBaseType: TypeAlias = Union[str, int, float, bool]
else:
    # The above will produce errors on isinstance if used in python 3.8. This
    # will work ok instead:
    TLensedBaseType = (str, int, float, bool)
"""Type of base types in span attributes.

!!! Warning
    OpenTelemetry does not allow None as an attribute value. Handling None is to
    be decided.
"""

TLensedAttributeValue = TypeAliasType(
    "TLensedAttributeValue",
    Union[
        str,
        int,
        float,
        bool,
        NoneType,  # TODO(SNOW-1711929): None is not technically allowed as an attribute value.
        Sequence["TLensedAttributeValue"],  # type: ignore
        "TLensedAttributes",
    ],
)
"""Type of values in span attributes."""

# NOTE(piotrm): pydantic will fail if you specify a recursive type alias without
# the TypeAliasType schema as above.

TLensedAttributes: TypeAlias = Dict[str, TLensedAttributeValue]
"""Attribute dictionaries.

Note that this deviates from what OTEL allows as attribute values. Because OTEL
does not allow general recursive values to be stored as attributes, we employ a
system of flattening values before exporting to OTEL. In this process we encode
a single generic value as multiple attributes where the attribute name include
paths/lenses to the parts of the generic value they are representing. For
example, an attribute/value like `{"a": {"b": 1, "c": 2}}` would be encoded as
`{"a.b": 1, "a.c": 2}`. This process is implemented in the
`flatten_lensed_attributes` method.
"""


def flatten_value(
    v: TLensedAttributeValue, lens: Optional[serial_utils.Lens] = None
) -> Iterable[Tuple[serial_utils.Lens, types_api.AttributeValue]]:
    """Flatten recursive value into OTEL-compatible attribute values.

    See `TLensedAttributes` for more details.
    """

    if lens is None:
        lens = serial_utils.Lens()

    # TODO(SNOW-1711929): OpenTelemetry does not allow None as an attribute
    # value. Unsure what is best to do here.

    # if v is None:
    #    yield (path, "None")

    elif v is None:
        pass

    elif isinstance(v, TLensedBaseType):
        yield (lens, v)

    elif isinstance(v, Sequence) and all(
        isinstance(e, TLensedBaseType) for e in v
    ):
        yield (lens, v)

    elif isinstance(v, Sequence):
        for i, e in enumerate(v):
            yield from flatten_value(v=e, lens=lens[i])

    elif isinstance(v, Mapping):
        for k, e in v.items():
            yield from flatten_value(v=e, lens=lens[k])

    else:
        raise ValueError(
            f"Do not know how to flatten value of type {type(v)} to OTEL attributes."
        )


def flatten_lensed_attributes(
    m: TLensedAttributes,
    path: Optional[serial_utils.Lens] = None,
    prefix: str = "",
) -> types_api.Attributes:
    """Flatten lensed attributes into OpenTelemetry attributes."""

    if path is None:
        path = serial_utils.Lens()

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
