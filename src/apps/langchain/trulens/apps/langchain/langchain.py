"""
Utilities for langchain apps. Includes component categories that organize
various langchain classes and example classes:
"""

from typing import Type

from trulens.core import app
from trulens.core.app import ComponentView
from trulens.core.utils.pyschema import Class
from trulens.core.utils.serial import JSON


class LangChainComponent(ComponentView):
    @staticmethod
    def class_is(cls_obj: Class) -> bool:
        if ComponentView.innermost_base(cls_obj.bases) == "langchain":
            return True

        return False

    @staticmethod
    def of_json(json: JSON) -> "LangChainComponent":
        return component_of_json(json)


class Prompt(app.Prompt, LangChainComponent):
    @property
    def template(self) -> str:
        return self.json["template"]

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(["template"]))

    @staticmethod
    def class_is(cls_obj: Class) -> bool:
        return cls_obj.noserio_issubclass(
            module_name="langchain.prompts.base",
            class_name="BasePromptTemplate",
        ) or cls_obj.noserio_issubclass(
            module_name="langchain.schema.prompt_template",
            class_name="BasePromptTemplate",
        )  # langchain >= 0.230


class LLM(app.LLM, LangChainComponent):
    @property
    def model_name(self) -> str:
        return self.json["model_name"]

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(["model_name"]))

    @staticmethod
    def class_is(cls_obj: Class) -> bool:
        return cls_obj.noserio_issubclass(
            module_name="langchain.llms.base", class_name="BaseLLM"
        )


class Other(app.Other, LangChainComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Prompt, LLM, Other]


def constructor_of_class(cls_obj: Class) -> Type[LangChainComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls_obj):
            return view

    raise TypeError(f"Unknown llama_index component type with class {cls_obj}")


def component_of_json(json: JSON) -> LangChainComponent:
    cls = Class.of_class_info(json)

    view = constructor_of_class(cls)

    return view(json)
