"""
Apps in trulens derive from two classes,
[AppVersionDefinition][trulens.core.schema.AppVersionDefinition] and
[App][trulens.core.app.App]. The first contains only serialized or serializable
components in a JSON-like format while the latter contains the executable apps
that may or may not be serializable.
"""

from trulens.core.app.base import AppVersion
from trulens.core.app.basic import TruBasicApp
from trulens.core.app.basic import TruWrapperApp
from trulens.core.app.custom import TruCustomApp
from trulens.core.app.virtual import TruVirtual
from trulens.core.app.virtual import VirtualApp

__all__ = [
    "AppVersion",
    "TruBasicApp",
    "TruWrapperApp",
    "TruCustomApp",
    "TruVirtual",
    "VirtualApp",
]
