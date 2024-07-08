"""
Utilities for app components provided as part of the trulens_eval package.
Currently organizes all such components as "Other".
"""

from typing import Type

from trulens_eval import app
from trulens_eval.utils.pyschema import Class


class Other(app.Other, app.TrulensComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Other]


def constructor_of_class(cls: Class) -> Type[app.TrulensComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls):
            return view

    raise TypeError(f"Unknown trulens component type with class {cls}")


def component_of_json(json: dict) -> app.TrulensComponent:
    cls = Class.of_class_info(json)

    view = constructor_of_class(cls)

    return view(json)
