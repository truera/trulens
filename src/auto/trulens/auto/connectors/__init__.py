# ruff: noqa: E402, F822

from typing import TYPE_CHECKING

from trulens.auto._utils import auto as auto_utils

if TYPE_CHECKING:
    # Needed for static tools to resolve submodules:
    from trulens.connectors.snowflake import SnowflakeDBConnector
    from trulens.core.database.connector.base import DefaultDBConnector

_CONNECTORS = {
    "DefaultDBConnector": (
        "trulens-core",
        "trulens.core.database.connector.base",
    ),
    "SnowflakeDBConnector": (
        "trulens-connector-snowflake",
        "trulens.connectors.snowflake",
    ),
}

_KINDS = {
    "connectors": _CONNECTORS,
}

__getattr__ = auto_utils.make_getattr_override(
    doc="TruLens database connectors.", kinds=_KINDS
)

__all__ = [
    "DefaultDBConnector",
    "SnowflakeDBConnector",
]
