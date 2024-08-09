"""
Apps in trulens derive from two classes,
[AppDefinition][trulens.core.schema.AppDefinition] and
[App][trulens.core.app.App]. The first contains only serialized or serializable
components in a JSON-like format while the latter contains the executable apps
that may or may not be serializable.
"""

from trulens.core.app.base import App
from trulens.core.app.basic import TruBasicApp
from trulens.core.app.basic import TruWrapperApp
from trulens.core.app.custom import TruCustomApp
from trulens.core.app.virtual import TruVirtual
from trulens.core.app.virtual import VirtualApp

__all__ = [
    "App",
    "TruBasicApp",
    "TruWrapperApp",
    "TruCustomApp",
    "TruVirtual",
    "VirtualApp",
]
