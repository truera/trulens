"""Context filters must keep the retriever's order, not judge finish order."""

import math
import threading
from unittest import mock

import pytest
from trulens.core.feedback import feedback as core_feedback
from trulens.core.guardrails import base as guardrails_base
from trulens.core.metric import metric as core_metric

_WAIT_SECONDS = 10.0


class _ReverseOrderJudge:
    """A judge whose calls finish in reverse input order.

    The call for `texts[i]` blocks until the call for `texts[i + 1]` has
    finished, so the last text always completes first. Scores come from
    `scores`, keyed by text.
    """

    def __init__(self, texts, scores):
        self.texts = list(texts)
        self.scores = scores
        self.done = {text: threading.Event() for text in self.texts}
        self.finished = []
        self.timed_out = []
        self._lock = threading.Lock()

    def evaluate(self, query, text):
        index = self.texts.index(text)
        if index + 1 < len(self.texts):
            later = self.done[self.texts[index + 1]]
            if not later.wait(_WAIT_SECONDS):
                self.timed_out.append(text)
        with self._lock:
            self.finished.append(text)
        self.done[text].set()
        return self.scores[text]

    def assert_finished_in_reverse(self):
        assert self.timed_out == []
        assert self.finished == list(reversed(self.texts))


def _filtered(judge, texts, higher_is_better=True):
    feedback = core_metric.Metric(
        implementation=judge.evaluate, higher_is_better=higher_is_better
    )

    @guardrails_base.context_filter(feedback, 0.5, "query")
    def retrieve(query):
        return list(texts)

    return retrieve("query")


def test_context_filter_keeps_input_order_when_later_contexts_finish_first():
    texts = ["c1", "c2", "c3", "c4"]
    judge = _ReverseOrderJudge(texts, dict.fromkeys(texts, 0.9))

    result = _filtered(judge, texts)

    judge.assert_finished_in_reverse()
    assert result == texts


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_context_filter_keeps_input_order_among_kept_contexts(
    higher_is_better,
):
    texts = ["keep1", "drop1", "keep2", "drop2", "keep3"]
    good, bad = (0.9, 0.1) if higher_is_better else (0.1, 0.9)
    scores = {text: good if text.startswith("keep") else bad for text in texts}
    judge = _ReverseOrderJudge(texts, scores)

    result = _filtered(judge, texts, higher_is_better=higher_is_better)

    judge.assert_finished_in_reverse()
    assert result == ["keep1", "keep2", "keep3"]


@pytest.mark.parametrize("invalid_score", [-1.0, math.nan])
def test_context_filter_drops_invalid_context_in_place(invalid_score, caplog):
    texts = ["first", "invalid", "middle", "last"]
    scores = dict.fromkeys(texts, 0.9)
    scores["invalid"] = invalid_score
    judge = _ReverseOrderJudge(texts, scores)

    result = _filtered(judge, texts)

    judge.assert_finished_in_reverse()
    assert result == ["first", "middle", "last"]
    assert "invalid guardrail score" in caplog.text


def _feedback(judge):
    return core_feedback.Feedback(judge.evaluate, higher_is_better=True)


@pytest.mark.optional
def test_langchain_filter_keeps_retrieval_order():
    lc_documents = pytest.importorskip("langchain_core.documents")
    lc_embeddings = pytest.importorskip("langchain_core.embeddings")
    lc_vectorstores = pytest.importorskip("langchain_core.vectorstores")
    lc_guardrails = pytest.importorskip("trulens.apps.langchain.guardrails")

    texts = ["keep1", "drop1", "keep2", "keep3"]
    scores = {text: 0.1 if text.startswith("drop") else 0.9 for text in texts}
    judge = _ReverseOrderJudge(texts, scores)
    retriever = lc_guardrails.WithFeedbackFilterDocuments(
        feedback=_feedback(judge),
        threshold=0.5,
        vectorstore=lc_vectorstores.InMemoryVectorStore(
            embedding=lc_embeddings.DeterministicFakeEmbedding(size=4)
        ),
    )
    docs = [lc_documents.Document(page_content=text) for text in texts]

    with mock.patch.object(
        lc_vectorstores.VectorStoreRetriever,
        "_get_relevant_documents",
        return_value=docs,
    ):
        result = retriever.invoke("query")

    judge.assert_finished_in_reverse()
    assert [doc.page_content for doc in result] == ["keep1", "keep2", "keep3"]


@pytest.mark.optional
def test_llamaindex_filter_keeps_retrieval_order():
    li_schema = pytest.importorskip("llama_index.core.schema")
    li_guardrails = pytest.importorskip("trulens.apps.llamaindex.guardrails")

    texts = ["keep1", "drop1", "keep2", "keep3"]
    scores = {text: 0.1 if text.startswith("drop") else 0.9 for text in texts}
    judge = _ReverseOrderJudge(texts, scores)
    query_engine = mock.MagicMock()
    query_engine.retrieve.return_value = [
        li_schema.NodeWithScore(node=li_schema.TextNode(text=text))
        for text in texts
    ]

    def synthesize(query_bundle, nodes, **kwargs):
        return nodes

    query_engine.synthesize.side_effect = synthesize
    engine = li_guardrails.WithFeedbackFilterNodes(
        query_engine, feedback=_feedback(judge), threshold=0.5
    )

    result = engine.query("query")

    judge.assert_finished_in_reverse()
    kept = [node.node.get_text() for node in result]
    assert kept == ["keep1", "keep2", "keep3"]
