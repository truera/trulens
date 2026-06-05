# pyright: reportMissingImports=false, reportMissingModuleSource=false
import pytest


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

    fake_scores = [{"label": "LABEL_0", "score": 0.1},
                   {"label": "LABEL_1", "score": 0.2},
                   {"label": "LABEL_2", "score": 0.7}]
    monkeypatch.setattr(Dummy, "_post_scores",
                        lambda self, **kwargs: fake_scores)

    result = provider.positive_sentiment(text="This is great!")
    assert isinstance(result, float)
    assert result == 0.7


@pytest.mark.optional
def test_toxic_returns_float(monkeypatch):
    from trulens.providers.huggingface.provider import Dummy
    provider = Dummy()

    fake_scores = [{"label": "toxic", "score": 0.9},
                   {"label": "non-toxic", "score": 0.1}]
    monkeypatch.setattr(Dummy, "_post_scores",
                        lambda self, **kwargs: fake_scores)

    result = provider.toxic(text="some text")
    assert isinstance(result, float)
    assert result == 0.9


@pytest.mark.optional
def test_context_relevance_returns_float(monkeypatch):
    from trulens.providers.huggingface.provider import Dummy
    provider = Dummy()

    fake_scores = [{"label": "context_relevance", "score": 0.85},
                   {"label": "other", "score": 0.15}]
    monkeypatch.setattr(Dummy, "_post_scores",
                        lambda self, **kwargs: fake_scores)

    result = provider.context_relevance(prompt="What is AI?",
                                        context="AI is artificial intelligence.")
    assert isinstance(result, float)
    assert result == 0.85


@pytest.mark.optional
def test_huggingface_local_model_loading_cached(monkeypatch):
    from trulens.providers.huggingface.provider import HuggingfaceLocal
    import trulens.providers.huggingface.provider as provider_mod

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