from typing import Iterable, List, Type

from trulens_eval import app
from trulens_eval.app import COMPONENT_CATEGORY
from trulens_eval.feedback import Feedback
from trulens_eval.util import Class
from trulens_eval.util import first
from trulens_eval.util import JSON
from trulens_eval.util import OptionalImports
from trulens_eval.util import second
from trulens_eval.util import TP


class Other(app.Other, app.TrulensComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Other]


def constructor_of_class(cls: Class) -> Type[app.TrulensComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls):
            return view

    raise TypeError(f"Unknown trulens component type with class {cls}")


def component_of_json(json: JSON) -> app.TrulensComponent:
    cls = Class.of_json(json)

    view = constructor_of_class(cls)

    return view(json)
