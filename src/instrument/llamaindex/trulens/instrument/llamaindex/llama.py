"""
Utilities for llama_index apps. Includes component categories that organize
various llama_index classes and example classes:

- `WithFeedbackFilterNodes`, a `VectorIndexRetriever` that filters retrieved
  nodes via a threshold on a specified feedback function.
"""

from typing import Type

from trulens.core import app
from trulens.core.app import ComponentView
from trulens.core.utils.pyschema import Class
from trulens.core.utils.serial import JSON


class LlamaIndexComponent(ComponentView):
    @staticmethod
    def class_is(cls_obj: Class) -> bool:
        if ComponentView.innermost_base(cls_obj.bases) == "llama_index":
            return True

        return False

    @staticmethod
    def of_json(json: JSON) -> "LlamaIndexComponent":
        return component_of_json(json)


class Prompt(app.Prompt, LlamaIndexComponent):
    @property
    def template(self) -> str:
        return self.json["template"]

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(["template"]))

    @staticmethod
    def class_is(cls_obj: Class) -> bool:
        return cls_obj.noserio_issubclass(
            module_name="llama_index.prompts.base", class_name="Prompt"
        )


class Agent(app.Agent, LlamaIndexComponent):
    @property
    def agent_name(self) -> str:
        return "agent name not supported in llama_index"

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set())

    @staticmethod
    def class_is(cls_obj: Class) -> bool:
        return cls_obj.noserio_issubclass(
            module_name="llama_index.agent.types", class_name="BaseAgent"
        )


class Tool(app.Tool, LlamaIndexComponent):
    @property
    def tool_name(self) -> str:
        if "metadata" in self.json:
            return self.json["metadata"]["name"]
        else:
            return "no name given"

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(["model"]))

    @staticmethod
    def class_is(cls_obj: Class) -> bool:
        return cls_obj.noserio_issubclass(
            module_name="llama_index.tools.types", class_name="BaseTool"
        )


class LLM(app.LLM, LlamaIndexComponent):
    @property
    def model_name(self) -> str:
        return self.json["model"]

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(["model"]))

    @staticmethod
    def class_is(cls_obj: Class) -> bool:
        return cls_obj.noserio_issubclass(
            module_name="llama_index.llms.base", class_name="LLM"
        )


class Other(app.Other, LlamaIndexComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Agent, Tool, Prompt, LLM, Other]


def constructor_of_class(cls_obj: Class) -> Type[LlamaIndexComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls_obj):
            return view

    raise TypeError(f"Unknown llama_index component type with class {cls_obj}")


def component_of_json(json: JSON) -> LlamaIndexComponent:
    cls = Class.of_class_info(json)

    view = constructor_of_class(cls)

    return view(json)
