# pyright: reportMissingImports=false, reportMissingModuleSource=false
import types
from unittest import mock

import pytest


class _FakeOutput(dict):
    def __init__(self, logits):
        super().__init__(logits=logits)
        self.logits = logits


def _fake_tokenizer_and_model(logits, id2label):
    import torch

    tokens = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    tokenizer = mock.Mock(return_value=tokens)
    tokenizer.decode.return_value = "Alice"
    model = mock.Mock(return_value=_FakeOutput(torch.tensor(logits)))
    model.config = types.SimpleNamespace(id2label=id2label)
    return tokenizer, model


@pytest.mark.optional
def test_huggingface_api_init():
    from trulens.providers.huggingface.provider import Dummy

    provider = Dummy()
    assert provider is not None


@pytest.mark.optional
def test_huggingface_local_init():
    from trulens.providers.huggingface.provider import HuggingfaceLocal

    provider = HuggingfaceLocal()
    assert provider is not None


@pytest.mark.optional
def test_tci_raises_on_empty_string():
    from trulens.providers.huggingface.provider import Dummy

    provider = Dummy()
    with pytest.raises(ValueError):
        provider.toxic(text="")


@pytest.mark.optional
def test_tci_raises_on_wrong_type():
    from trulens.providers.huggingface.provider import Dummy

    provider = Dummy()
    with pytest.raises(TypeError):
        provider.language_match(text1="hello", text2=123)


@pytest.mark.optional
def test_positive_sentiment_returns_float(monkeypatch):
    from trulens.providers.huggingface.provider import Dummy

    provider = Dummy()

    fake_scores = [
        {"label": "LABEL_0", "score": 0.1},
        {"label": "LABEL_1", "score": 0.2},
        {"label": "LABEL_2", "score": 0.7},
    ]
    monkeypatch.setattr(
        Dummy, "_post_scores", lambda self, **kwargs: fake_scores
    )

    result = provider.positive_sentiment(text="This is great!")
    assert isinstance(result, float)
    assert result == 0.7


@pytest.mark.optional
def test_toxic_returns_float(monkeypatch):
    from trulens.providers.huggingface.provider import Dummy

    provider = Dummy()

    fake_scores = [
        {"label": "toxic", "score": 0.9},
        {"label": "non-toxic", "score": 0.1},
    ]
    monkeypatch.setattr(
        Dummy, "_post_scores", lambda self, **kwargs: fake_scores
    )

    result = provider.toxic(text="some text")
    assert isinstance(result, float)
    assert result == 0.9


@pytest.mark.optional
def test_context_relevance_returns_float(monkeypatch):
    from trulens.providers.huggingface.provider import Dummy

    provider = Dummy()

    fake_scores = [
        {"label": "context_relevance", "score": 0.85},
        {"label": "other", "score": 0.15},
    ]
    monkeypatch.setattr(
        Dummy, "_post_scores", lambda self, **kwargs: fake_scores
    )

    result = provider.context_relevance(
        prompt="What is AI?", context="AI is artificial intelligence."
    )
    assert isinstance(result, float)
    assert result == 0.85


@pytest.mark.optional
def test_huggingface_local_model_loading_cached(monkeypatch):
    import trulens.providers.huggingface.provider as provider_mod
    from trulens.providers.huggingface.provider import HuggingfaceLocal

    provider = HuggingfaceLocal()

    # Clear instance-level cache before test
    provider._cached_tokenizers.clear()
    provider._cached_models.clear()

    tokenizer_call_count = [0]
    model_call_count = [0]

    def fake_tokenizer_load(key, **kwargs):
        tokenizer_call_count[0] += 1
        return object()

    def fake_model_load(key):
        model_call_count[0] += 1
        return object()

    monkeypatch.setattr(
        provider_mod.AutoTokenizer, "from_pretrained", fake_tokenizer_load
    )
    monkeypatch.setattr(
        provider_mod.AutoModelForSequenceClassification,
        "from_pretrained",
        fake_model_load,
    )

    # First call — should load from pretrained
    tokenizer1, model1 = provider._retrieve_tokenizer_and_model("some-model")
    assert tokenizer_call_count[0] == 1
    assert model_call_count[0] == 1

    # Second call with same key — should use cache, not reload
    tokenizer2, model2 = provider._retrieve_tokenizer_and_model("some-model")
    assert tokenizer_call_count[0] == 1
    assert model_call_count[0] == 1

    # Same objects returned from cache
    assert tokenizer1 is tokenizer2
    assert model1 is model2


@pytest.mark.optional
def test_huggingface_local_language_scores(monkeypatch):
    from trulens.providers.huggingface.provider import HuggingfaceLocal

    provider = HuggingfaceLocal()
    tokenizer, model = _fake_tokenizer_and_model(
        [[0.2, 1.8]], {0: "en", 1: "fr"}
    )
    monkeypatch.setattr(
        provider,
        "_retrieve_tokenizer_and_model",
        mock.Mock(return_value=(tokenizer, model)),
    )

    result = provider._language_scores_endpoint("bonjour")

    assert set(result) == {"en", "fr"}
    assert all(isinstance(score, float) for score in result.values())
    model.assert_called_once_with(**tokenizer.return_value)


@pytest.mark.optional
@pytest.mark.parametrize(
    ("endpoint_name", "expected_label"),
    [
        ("_context_relevance_endpoint", "context_relevance"),
        ("_positive_sentiment_endpoint", "LABEL_2"),
        ("_toxic_endpoint", "toxic"),
        ("_summarized_groundedness_endpoint", "entailment"),
    ],
)
def test_huggingface_local_scalar_endpoints(
    monkeypatch, endpoint_name, expected_label
):
    from trulens.providers.huggingface.provider import HuggingfaceLocal

    provider = HuggingfaceLocal()
    tokenizer, model = _fake_tokenizer_and_model(
        [[0.2, 1.8]], {0: "other", 1: expected_label}
    )
    monkeypatch.setattr(
        provider,
        "_retrieve_tokenizer_and_model",
        mock.Mock(return_value=(tokenizer, model)),
    )

    result = getattr(provider, endpoint_name)("input")

    assert isinstance(result, float)
    model.assert_called_once_with(**tokenizer.return_value)


@pytest.mark.optional
@pytest.mark.parametrize(
    ("endpoint_name", "expected_label"),
    [
        ("_context_relevance_endpoint", "context_relevance"),
        ("_positive_sentiment_endpoint", "LABEL_2"),
        ("_toxic_endpoint", "toxic"),
        ("_summarized_groundedness_endpoint", "entailment"),
    ],
)
def test_huggingface_local_label_errors_list_available_labels(
    monkeypatch, endpoint_name, expected_label
):
    from trulens.providers.huggingface.provider import HuggingfaceLocal

    provider = HuggingfaceLocal()
    tokenizer, model = _fake_tokenizer_and_model(
        [[0.2, 1.8]], {0: "negative", 1: "positive"}
    )
    monkeypatch.setattr(
        provider,
        "_retrieve_tokenizer_and_model",
        mock.Mock(return_value=(tokenizer, model)),
    )

    with pytest.raises(RuntimeError) as exc_info:
        getattr(provider, endpoint_name)("input")

    assert expected_label in str(exc_info.value)
    assert "['negative', 'positive']" in str(exc_info.value)


@pytest.mark.optional
def test_huggingface_local_pii_detection(monkeypatch):
    from trulens.providers.huggingface.provider import HuggingfaceLocal

    provider = HuggingfaceLocal()
    tokenizer, model = _fake_tokenizer_and_model(
        [[[0.1, 1.9], [1.9, 0.1]]], {0: "O", 1: "B-NAME"}
    )
    monkeypatch.setattr(
        provider,
        "_retrieve_tokenizer_and_model",
        mock.Mock(return_value=(tokenizer, model)),
    )

    result = provider._pii_detection_endpoint("Alice")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], float)
    model.assert_called_once_with(**tokenizer.return_value)


@pytest.mark.optional
def test_huggingface_local_pii_detection_with_reasons(monkeypatch):
    from trulens.providers.huggingface.provider import HuggingfaceLocal

    provider = HuggingfaceLocal()
    tokenizer, model = _fake_tokenizer_and_model(
        [[[0.1, 1.9], [1.9, 0.1]]], {0: "O", 1: "B-NAME"}
    )
    monkeypatch.setattr(
        provider,
        "_retrieve_tokenizer_and_model",
        mock.Mock(return_value=(tokenizer, model)),
    )

    scores, reasons = provider._pii_detection_with_cot_reasons_endpoint("Alice")

    assert isinstance(scores, list)
    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert list(reasons) == ["NAME detected: Alice"]
    model.assert_called_once_with(**tokenizer.return_value)


@pytest.mark.optional
def test_huggingface_local_hallucination_endpoint(monkeypatch):
    import trulens.providers.huggingface.provider as provider_mod

    provider = provider_mod.HuggingfaceLocal()
    provider._cached_tokenizers.clear()
    provider._cached_models.clear()
    tokenizer, model = _fake_tokenizer_and_model([[0.2, 1.8]], {})
    tokenizer_loader = mock.Mock(return_value=tokenizer)
    model_loader = mock.Mock(return_value=model)
    monkeypatch.setattr(
        provider_mod.AutoTokenizer, "from_pretrained", tokenizer_loader
    )
    monkeypatch.setattr(
        provider_mod.AutoModelForSequenceClassification,
        "from_pretrained",
        model_loader,
    )

    result = provider._hallucination_evaluator_endpoint("premise hypothesis")

    assert isinstance(result, float)
    tokenizer_loader.assert_called_once_with("google/flan-t5-base")
    model_loader.assert_called_once_with(
        provider_mod.HUGS_HALLUCINATION_MODEL_PATH,
        trust_remote_code=True,
    )
    model.assert_called_once_with(**tokenizer.return_value)


@pytest.mark.optional
def test_feedback_raises_on_empty_scores(monkeypatch):
    from trulens.providers.huggingface.provider import Dummy

    provider = Dummy()

    monkeypatch.setattr(Dummy, "_post_scores", lambda self, **kwargs: [])

    with pytest.raises(RuntimeError):
        provider.positive_sentiment(text="some text")
