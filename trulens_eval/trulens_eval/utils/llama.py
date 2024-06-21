"""
Utilities for llama_index apps. Includes component categories that organize
various llama_index classes and example classes:

- `WithFeedbackFilterNodes`, a `VectorIndexRetriever` that filters retrieved
  nodes via a threshold on a specified feedback function.
"""

from typing import Type

from trulens_eval import app
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LLAMA
from trulens_eval.utils.pyschema import Class

with OptionalImports(messages=REQUIREMENT_LLAMA) as opt:
    import llama_index

opt.assert_installed(llama_index)


class Prompt(app.Prompt, app.LlamaIndexComponent):

    @property
    def template(self) -> str:
        return self.json['template']

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(['template']))

    @staticmethod
    def class_is(cls: Class) -> bool:
        return cls.noserio_issubclass(
            module_name="llama_index.prompts.base", class_name="Prompt"
        )


class Agent(app.Agent, app.LlamaIndexComponent):

    @property
    def agent_name(self) -> str:
        return "agent name not supported in llama_index"

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set())

    @staticmethod
    def class_is(cls: Class) -> bool:
        return cls.noserio_issubclass(
            module_name="llama_index.agent.types", class_name="BaseAgent"
        )


class Tool(app.Tool, app.LlamaIndexComponent):

    @property
    def tool_name(self) -> str:
        if 'metadata' in self.json:
            return self.json['metadata']['name']
        else:
            return "no name given"

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(['model']))

    @staticmethod
    def class_is(cls: Class) -> bool:
        return cls.noserio_issubclass(
            module_name="llama_index.tools.types", class_name="BaseTool"
        )


class LLM(app.LLM, app.LlamaIndexComponent):

    @property
    def model_name(self) -> str:
        return self.json['model']

    def unsorted_parameters(self):
        return super().unsorted_parameters(skip=set(['model']))

    @staticmethod
    def class_is(cls: Class) -> bool:
        return cls.noserio_issubclass(
            module_name="llama_index.llms.base", class_name="LLM"
        )


class Other(app.Other, app.LlamaIndexComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Agent, Tool, Prompt, LLM, Other]


def constructor_of_class(cls: Class) -> Type[app.LlamaIndexComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls):
            return view

    raise TypeError(f"Unknown llama_index component type with class {cls}")


def component_of_json(json: dict) -> app.LlamaIndexComponent:
    cls = Class.of_class_info(json)

    view = constructor_of_class(cls)

    return view(json)
