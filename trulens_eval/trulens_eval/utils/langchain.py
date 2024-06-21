"""
Utilities for langchain apps. Includes component categories that organize
various langchain classes and example classes:
"""
from typing import Type

from trulens_eval import app
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LANGCHAIN
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.serial import JSON

with OptionalImports(messages=REQUIREMENT_LANGCHAIN):
    import langchain


class Prompt(app.Prompt, app.LangChainComponent):

    @property
    def template(self) -> str:
        return self.json['template']

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(['template']))

    @staticmethod
    def class_is(cls: Class) -> bool:
        return cls.noserio_issubclass(
            module_name="langchain.prompts.base",
            class_name="BasePromptTemplate"
        ) or cls.noserio_issubclass(
            module_name="langchain.schema.prompt_template",
            class_name="BasePromptTemplate"
        )  # langchain >= 0.230


class LLM(app.LLM, app.LangChainComponent):

    @property
    def model_name(self) -> str:
        return self.json['model_name']

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(['model_name']))

    @staticmethod
    def class_is(cls: Class) -> bool:
        return cls.noserio_issubclass(
            module_name="langchain.llms.base", class_name="BaseLLM"
        )


class Other(app.Other, app.LangChainComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Prompt, LLM, Other]


def constructor_of_class(cls: Class) -> Type[app.LangChainComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls):
            return view

    raise TypeError(f"Unknown llama_index component type with class {cls}")


def component_of_json(json: JSON) -> app.LangChainComponent:
    cls = Class.of_class_info(json)

    view = constructor_of_class(cls)

    return view(json)
