"""
Utilities for langchain apps. Includes component categories that organize
various langchain classes and example classes:

- `WithFeedbackFilterDocuments`: a `VectorStoreRetriever` that filters retrieved
  documents via a threshold on a specified feedback function.
"""

from concurrent.futures import wait
from typing import List, Type

from trulens_eval import app
from trulens_eval.feedback import Feedback
from trulens_eval.utils.containers import first
from trulens_eval.utils.containers import second
from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LANGCHAIN
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import model_dump
from trulens_eval.utils.threading import ThreadPoolExecutor

with OptionalImports(messages=REQUIREMENT_LANGCHAIN):
    import langchain
    from langchain.schema import Document
    from langchain.vectorstores.base import VectorStoreRetriever


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


class WithFeedbackFilterDocuments(VectorStoreRetriever):
    feedback: Feedback
    threshold: float

    def __init__(self, feedback: Feedback, threshold: float, *args, **kwargs):
        """
        A VectorStoreRetriever that filters documents using a minimum threshold
        on a feedback function before returning them.

        - feedback: Feedback - use this feedback function to score each
          document.
        
        - threshold: float - and keep documents only if their feedback value is
          at least this threshold.
        """

        super().__init__(
            *args, feedback=feedback, threshold=threshold, **kwargs
        )

    # Signature must match
    # langchain.schema.retriever.BaseRetriever._get_relevant_documents .
    def _get_relevant_documents(self, query: str, *,
                                run_manager) -> List[Document]:
        # Get relevant docs using super class:
        docs = super()._get_relevant_documents(query, run_manager=run_manager)

        # Evaluate the filter on each, in parallel.
        ex = ThreadPoolExecutor(max_workers=max(1, len(docs)))

        futures = list(
            (
                doc,
                ex.submit(
                    (
                        lambda doc, query: self.
                        feedback(query, doc.page_content) > self.threshold
                    ),
                    query=query,
                    doc=doc
                )
            ) for doc in docs
        )

        wait([future for (_, future) in futures])

        results = list((doc, future.result()) for (doc, future) in futures)
        filtered = map(first, filter(second, results))

        # Return only the filtered ones.
        return list(filtered)

    @staticmethod
    def of_retriever(retriever: VectorStoreRetriever, **kwargs):
        return WithFeedbackFilterDocuments(**kwargs, **(model_dump(retriever)))
