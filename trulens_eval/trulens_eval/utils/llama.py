from typing import Iterable

from trulens_eval.tru_app import COMPONENT_CATEGORY
from trulens_eval.util import Class
from trulens_eval.util import OptionalImports
from trulens_eval.util import REQUIREMENT_LLAMA
from trulens_eval.util import TP

with OptionalImports(message=REQUIREMENT_LLAMA):
    import llama_index


class Is:
    """
    Various checks for typical llama index components based on their names (i.e.
    without actually loading them). See util.py:WithClassInfo for more.
    """

    @staticmethod
    def engine(cls: Class):
        return cls.noserio_issubclass(
            module_name="llama_index.indices.query.base",
            class_name="BaseQueryEngine"
        )

    @staticmethod
    def retriever(cls: Class):
        return cls.noserio_issubclass(
            module_name="llama_index.indices.base_retriever",
            class_name="BaseRetriever"
        )

    @staticmethod
    def selector(cls: Class):
        return cls.noserio_issubclass(
            module_name="llama_index.selectors.types",
            class_name="BaseSelector"
        )

    @staticmethod
    def what(cls: Class) -> Iterable[COMPONENT_CATEGORY]:
        CHECKERS = [Is.engine, Is.retriever, Is.selector]

        for checker in CHECKERS:
            if checker(cls):
                yield checker.__name__


# TODO: same for llama index:
# class WithFeedbackFilterDocuments(VectorStoreRetriever):
#feedback: Feedback
#threshold: float
