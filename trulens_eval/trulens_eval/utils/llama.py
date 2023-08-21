"""
Utilities for llama_index apps. Includes component categories that organize
various llama_index classes and example classes:

- `WithFeedbackFilterNodes`, a `VectorIndexRetriever` that filters retrieved
  nodes via a threshold on a specified feedback function.
"""

from typing import List, Type

from trulens_eval import app
from trulens_eval import Feedback
from trulens_eval.feedback import Feedback
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.containers import first
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LLAMA
from trulens_eval.utils.containers import second
from trulens_eval.utils.threading import TP

with OptionalImports(message=REQUIREMENT_LLAMA):
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.indices.vector_store.retrievers import \
        VectorIndexRetriever
    from llama_index.schema import NodeWithScore


class Prompt(app.Prompt, app.LangChainComponent):

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


class Other(app.Other, app.LlamaIndexComponent):
    pass


# All component types, keep Other as the last one since it always matches.
COMPONENT_VIEWS = [Prompt, Other]


def constructor_of_class(cls: Class) -> Type[app.LlamaIndexComponent]:
    for view in COMPONENT_VIEWS:
        if view.class_is(cls):
            return view

    raise TypeError(f"Unknown llama_index component type with class {cls}")


def component_of_json(json: JSON) -> app.LlamaIndexComponent:
    cls = Class.of_json(json)

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

        # Evaluate the filter on each, in parallel.
        promises = (
            (
                node, TP().promise(
                    lambda query, node: self.feedback(
                        query.query_str, node.node.get_text()
                    ) > self.threshold,
                    query=query_bundle,
                    node=node
                )
            ) for node in nodes
        )
        results = ((node, promise.get()) for (node, promise) in promises)
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
