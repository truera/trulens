"""
Utilities for app components provided as part of the trulens package.
Currently organizes all such components as "Other".
"""

from trulens.core import app
from trulens.core._utils.pycompat import Type
from trulens.core.utils import pyschema as pyschema_utils


class Other(app.Other, app.TrulensComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Other]


def constructor_of_class(
    cls_obj: pyschema_utils.Class,
) -> Type[app.TrulensComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls_obj):
            return view

    raise TypeError(f"Unknown trulens component type with class {cls_obj}")


def component_of_json(json: dict) -> app.TrulensComponent:
    cls = pyschema_utils.Class.of_class_info(json)

    view = constructor_of_class(cls)

    return view(json)
