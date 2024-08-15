"""
Utilities for app components provided as part of the trulens package.
Currently organizes all such components as "Other".
"""

import time
from typing import Type

import pandas as pd
from trulens.core.app import base
from trulens.core.schema.record import Record
from trulens.core.utils.pyschema import Class


class Other(base.Other, base.TrulensComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Other]


def constructor_of_class(cls_obj: Class) -> Type[base.TrulensComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls_obj):
            return view

    raise TypeError(f"Unknown trulens component type with class {cls_obj}")


def component_of_json(json: dict) -> base.TrulensComponent:
    cls = Class.of_class_info(json)

    view = constructor_of_class(cls)

    return view(json)