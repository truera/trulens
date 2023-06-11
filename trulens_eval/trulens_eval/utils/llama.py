from typing import Callable, Iterable, List
from pydantic import Field
from trulens_eval.tru_feedback import Feedback
from trulens_eval.tru_model import COMPONENT_CATEGORY
from trulens_eval.util import Class
from trulens_eval.util import TP, first, second



class Is:
    """
    Various checks for typical llama index components based on their names (i.e.
    without actually loading them). See util.py:WithClassInfo for more.
    """

    @staticmethod
    def engine(cls: Class):
        return cls.noserio_issubclass(
            module_name="llama_index.query_engine.retriever_query_engine", class_name="RetrieverQueryEngine"
        )

    @staticmethod
    def what(cls: Class) -> Iterable[COMPONENT_CATEGORY]:
        CHECKERS = [
            Is.engine
        ]

        for checker in CHECKERS:
            if checker(cls):
                yield checker.__name__


# TODO: same for llama index:
# class WithFeedbackFilterDocuments(VectorStoreRetriever):
    #feedback: Feedback
    #threshold: float

  