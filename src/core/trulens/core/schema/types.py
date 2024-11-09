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
import random
import time
from typing import Dict, Optional, Tuple, Type, Union
import uuid

from opentelemetry import trace as trace_api
from sqlalchemy import BINARY
from sqlalchemy import JSON
from sqlalchemy import TIMESTAMP
from sqlalchemy import VARCHAR
from trulens.core._utils.pycompat import TypeAlias  # import style exception

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


class TypeInfo:
    """Container for types and conversions between types used to represent
    values that need to be stored/represented in different places.

    The places are at least python and sql and optionally in OTEL.

    The capabilities this class exposes are:

    - Type aliases in the form of `{PY,SQL,OTEL}_TYPE`.

    - Type union for the above.

    -
    """

    OTEL_TYPE: Optional[Type] = None
    """Type for OTEL values.

    None if type info is not relevant to OTEL.
    """

    NUM_BITS: Optional[int] = None
    """Number of bits in values of this type.

    Note that when represented in python, OTEL, or SQL, the number of bits may
    vary but may be specified by a spec, like OTEL, to be a certain number of
    bits regardless."""

    PY_TYPE: Type
    """Type for python values."""

    SQL_TYPE: Type
    """Type for values understood by sqlalchemy to be representing in the
    database as the column type `SQL_SCHEMA_TYPE`."""

    SQL_SCHEMA_TYPE: Type
    """SQL column type for the column type."""

    UNION_TYPE: Type
    """Union of all types that can be used to represent values of this type
    except the schema type."""

    TYPES: Tuple[Type, ...]
    """Tuple of the above so that isinstance can be used."""

    @classmethod
    def py(cls, val: TypeInfo.UNION_TYPE) -> TypeInfo.PY_TYPE:
        """Convert a value to a python value."""
        if isinstance(val, cls.PY_TYPE):
            return val
        if isinstance(val, cls.SQL_TYPE):
            return cls.py_of_sql(val)
        if cls.OTEL_TYPE is not None and isinstance(val, cls.OTEL_TYPE):
            return cls.py_of_otel(val)

        raise TypeError(f"Cannot convert value of type {type(val)} to python.")

    @classmethod
    def otel(cls, val: TypeInfo.UNION_TYPE) -> TypeInfo.OTEL_TYPE:
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
    def sql(cls, val: TypeInfo.UNION_TYPE) -> TypeInfo.SQL_TYPE:
        """Convert a value to the sql representation."""

        if isinstance(val, cls.SQL_TYPE):
            return val
        if isinstance(val, cls.PY_TYPE):
            return cls.sql_of_py(val)
        if cls.OTEL_TYPE is not None and isinstance(val, cls.OTEL_TYPE):
            return cls.sql_of_otel(val)

        raise TypeError(f"Cannot convert value of type {type(val)} to sql.")

    @classmethod
    def default_py(cls) -> TypeInfo.PY_TYPE:
        """Default python value for this type."""
        return cls.rand_py()

    @classmethod
    def default_sql(cls) -> TypeInfo.SQL_TYPE:
        """Default sql value for this type."""
        return cls.rand_sql()

    @classmethod
    def rand_py(cls) -> TypeInfo.PY_TYPE:
        """Generate a new random python value of this type."""
        if cls.rand_otel is not TypeInfo.rand_otel:
            return cls.py_of_otel(cls.rand_otel())
        if cls.rand_sql is not TypeInfo.rand_sql:
            return cls.py_of_sql(cls.rand_sql())
        raise NotImplementedError("Python type generation not implemented.")

    @classmethod
    def rand_sql(cls) -> TypeInfo.SQL_TYPE:
        """Generate a new random sql value of this type."""
        if cls.rand_otel is not TypeInfo.rand_otel:
            return cls.sql_of_otel(cls.rand_otel())
        if cls.rand_py is not TypeInfo.rand_py:
            return cls.sql_of_py(cls.rand_py())

        raise NotImplementedError("SQL type generation not implemented.")

    @classmethod
    def sql_of_py(cls, py_value: TypeInfo.PY_TYPE) -> TypeInfo.SQL_TYPE:
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
    def py_of_sql(cls, sql_value: TypeInfo.SQL_TYPE) -> TypeInfo.PY_TYPE:
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
    def rand_otel(cls) -> TypeInfo.OTEL_TYPE:
        """Generate a new random otel value of this type."""

        cls._assert_has_otel()

        if cls.rand_py is not TypeInfo.rand_py:
            return cls.otel_of_py(cls.rand_py())
        if cls.rand_sql is not TypeInfo.rand_sql:
            return cls.otel_of_sql(cls.rand_sql())

        raise NotImplementedError("OTEL type generation not implemented.")

    @classmethod
    def default_otel(cls) -> TypeInfo.OTEL_TYPE:
        """Default otel value for this type."""

        cls._assert_has_otel()

        return cls.rand_otel()

    @classmethod
    def otel_of_py(cls, py_value: TypeInfo.PY_TYPE) -> TypeInfo.OTEL_TYPE:
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
    def py_of_otel(cls, otel_value: TypeInfo.OTEL_TYPE) -> TypeInfo.PY_TYPE:
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
    def otel_of_sql(cls, sql_value: TypeInfo.SQL_TYPE) -> TypeInfo.OTEL_TYPE:
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
    def sql_of_otel(cls, otel_value: TypeInfo.OTEL_TYPE) -> TypeInfo.SQL_TYPE:
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


class SpanID(TypeInfo):
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

    @classmethod
    def rand_otel(cls) -> int:
        return int(
            random.getrandbits(cls.NUM_BITS) & trace_api.span._SPAN_ID_MAX_VALUE
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
        return py_value.to_bytes(cls.NUM_BITS // 8)

    @classmethod
    def py_of_sql(cls, sql_value: bytes) -> int:
        return int.from_bytes(sql_value)


class TraceID(TypeInfo):
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
        return py_value.to_bytes(cls.NUM_BITS // 8)

    @classmethod
    def py_of_sql(cls, sql_value: bytes) -> int:
        return int.from_bytes(sql_value)


class TraceRecordID(TypeInfo):
    """Types for representing record ids in traces/spans."""

    OTEL_TYPE: Type = str  # same as RecordId above
    PY_TYPE: Type = str  # same as RecordId above
    SQL_TYPE: Type = str  # same as RecordId above
    SQL_SCHEMA_TYPE: Type = VARCHAR(256)  # same as orm.py:TYPE_ID

    UNION_TYPE = str
    TYPES = (str,)

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


class TraceRecordIDs(TypeInfo):
    """Type for representing multiple trace record ids.

    This is a list of trace record ids. It is a list of `TraceRecordID`.
    """

    OTEL_TYPE: Type = list
    PY_TYPE: Type = list
    SQL_TYPE: Type = list
    SQL_SCHEMA_TYPE: Type = JSON

    @classmethod
    def otel_of_py(cls, py_value: list) -> list:
        return [TraceRecordID.otel_of_py(val) for val in py_value]

    @classmethod
    def py_of_otel(cls, otel_value: list) -> list:
        return [TraceRecordID.py_of_otel(val) for val in otel_value]

    @classmethod
    def sql_of_py(cls, py_value: list) -> list:
        return [TraceRecordID.sql_of_py(val) for val in py_value]

    @classmethod
    def py_of_sql(cls, sql_value: list) -> list:
        return [TraceRecordID.py_of_sql(val) for val in sql_value]


class Timestamp(TypeInfo):
    """Timestamp type.

    This type is for supporting OTEL hence its requirements come from there. In
    OTEL it is a 64-bit integer representing the number of nano seconds since
    the epoch. In python, it is a [datetime][datetime.datetime]. In the
    database, it is the TIMESTAMP sql column type.
    """

    NUM_BITS: int = 64
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
    def rand_py(cls) -> int:
        raise NotImplementedError("Timestamps are not meant to be random.")

    @classmethod
    def rand_sql(cls) -> bytes:
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
