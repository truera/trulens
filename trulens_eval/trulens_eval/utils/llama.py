"""
Utilities for llama_index apps. Includes component categories that organize
various llama_index classes and example classes:

- `WithFeedbackFilterNodes`, a `VectorIndexRetriever` that filters retrieved
  nodes via a threshold on a specified feedback function.
"""

from typing import List, Type

from trulens_eval import app
from trulens_eval.feedback import Feedback
from trulens_eval.utils.containers import first
from trulens_eval.utils.containers import second
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LLAMA
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.threading import ThreadPoolExecutor

with OptionalImports(messages=REQUIREMENT_LLAMA):
    import llama_index
    from llama_index.core.indices.vector_store.retrievers.retriever import \
        VectorIndexRetriever
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.schema import NodeWithScore

OptionalImports(messages=REQUIREMENT_LLAMA).assert_installed(llama_index)


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


class WithFeedbackFilterNodes(VectorIndexRetriever):
    feedback: Feedback
    threshold: float

    def __init__(self, feedback: Feedback, threshold: float, *args, **kwargs):
        """
        A VectorIndexRetriever that filters documents using a minimum threshold
        on a feedback function before returning them.

        - feedback: Feedback - use this feedback function to score each
        document.
        
        - threshold: float - and keep documents only if their feedback value is
        at least this threshold.
        """

        super().__init__(*args, **kwargs)

        self.feedback = feedback
        self.threshold = threshold

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Get relevant docs using super class:
        nodes = super()._retrieve(query_bundle)

        ex = ThreadPoolExecutor(max_workers=max(1, len(nodes)))

        # Evaluate the filter on each, in parallel.
        futures = (
            (
                node,
                ex.submit(
                    lambda query, node: self.feedback(
                        query.query_str, node.node.get_text()
                    ) > self.threshold,
                    query=query_bundle,
                    node=node
                )
            ) for node in nodes
        )

        wait([future for (_, future) in futures])

        results = ((node, future.result()) for (node, future) in futures)
        filtered = map(first, filter(second, results))

        # Return only the filtered ones.
        return list(filtered)

    @staticmethod
    def of_index_retriever(retriever: VectorIndexRetriever, **kwargs):
        return WithFeedbackFilterNodes(
            index=retriever._index,
            similarty_top_k=retriever._similarity_top_k,
            vectore_store_query_mode=retriever._vector_store_query_mode,
            filters=retriever._filters,
            alpha=retriever._alpha,
            doc_ids=retriever._doc_ids,
            **kwargs
        )
