"""Integration context filters reject failed scores in either direction."""

import math
from unittest import mock

import pytest
from trulens.core.feedback import feedback as core_feedback
from trulens.core.utils import constants as constants_utils

pytestmark = pytest.mark.optional


@pytest.fixture(params=["langchain", "llamaindex"])
def filter_contexts(request):
    """Exercise each integration's real filtering entry point."""
    if request.param == "langchain":
        documents = pytest.importorskip("langchain_core.documents")
        embeddings = pytest.importorskip("langchain_core.embeddings")
        vectorstores = pytest.importorskip("langchain_core.vectorstores")
        guardrails = pytest.importorskip("trulens.apps.langchain.guardrails")

        def run(feedback, texts):
            retriever = guardrails.WithFeedbackFilterDocuments(
                feedback=feedback,
                threshold=0.5,
                vectorstore=vectorstores.InMemoryVectorStore(
                    embedding=embeddings.DeterministicFakeEmbedding(size=4)
                ),
            )
            docs = [documents.Document(page_content=text) for text in texts]
            with mock.patch.object(
                vectorstores.VectorStoreRetriever,
                "_get_relevant_documents",
                return_value=docs,
            ):
                return [doc.page_content for doc in retriever.invoke("query")]

    else:
        schema = pytest.importorskip("llama_index.core.schema")
        guardrails = pytest.importorskip("trulens.apps.llamaindex.guardrails")

        def run(feedback, texts):
            engine = mock.MagicMock()
            engine.retrieve.return_value = [
                schema.NodeWithScore(node=schema.TextNode(text=text))
                for text in texts
            ]
            engine.synthesize.side_effect = (
                lambda query_bundle, nodes, **kwargs: nodes
            )
            filtered = guardrails.WithFeedbackFilterNodes(
                engine, feedback=feedback, threshold=0.5
            ).query("query")
            return [node.node.get_text() for node in filtered]

    return run


def _feedback(scores, higher_is_better):
    def evaluate(query, text):
        return scores[text]

    return core_feedback.Feedback(evaluate, higher_is_better=higher_is_better)


@pytest.mark.parametrize("higher_is_better", [True, False])
@pytest.mark.parametrize(
    "invalid_score",
    [constants_utils.UNPARSABLE_SCORE, math.nan, math.inf, -math.inf],
)
def test_invalid_context_is_excluded_without_losing_valid_neighbors(
    filter_contexts, higher_is_better, invalid_score, caplog
):
    texts = ["first", "invalid", "irrelevant", "last"]
    good, bad = (0.8, 0.2) if higher_is_better else (0.2, 0.8)
    scores = {
        "first": good,
        "invalid": invalid_score,
        "irrelevant": bad,
        "last": good,
    }

    kept = filter_contexts(_feedback(scores, higher_is_better), texts)

    assert kept == ["first", "last"]
    assert "invalid guardrail score" in caplog.text


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_entirely_ungraded_contexts_are_excluded(
    filter_contexts, higher_is_better
):
    scores = {
        "unparsable": constants_utils.UNPARSABLE_SCORE,
        "nan": math.nan,
        "infinity": math.inf,
        "negative infinity": -math.inf,
    }

    assert (
        filter_contexts(_feedback(scores, higher_is_better), list(scores)) == []
    )
