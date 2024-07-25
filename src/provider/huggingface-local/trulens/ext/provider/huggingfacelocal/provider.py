import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from trulens.ext.provider.huggingface.provider import HUGS_DOCNLI_MODEL_PATH
from trulens.ext.provider.huggingface.provider import HuggingfaceBase
from trulens.ext.provider.huggingface.provider import _tci

logger = logging.getLogger(__name__)


class HuggingfaceLocal(HuggingfaceBase):
    """
    Out of the box feedback functions using HuggingFace models locally.
    """

    _cached_tokenizers: Dict[str, Any] = {}
    _cached_models: Dict[str, Any] = {}

    def _retrieve_tokenizer_and_model(
        self, key: str, tokenizer_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Any]:
        if key not in self._cached_tokenizers:
            tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs else {}
            self._cached_tokenizers[key] = AutoTokenizer.from_pretrained(
                key, **tokenizer_kwargs
            )
        if key not in self._cached_models:
            self._cached_models[key] = (
                AutoModelForSequenceClassification.from_pretrained(key)
            )
        tokenizer = self._cached_tokenizers[key]
        model = self._cached_models[key]
        return tokenizer, model

    def _language_scores_endpoint(self, text: str) -> Dict[str, float]:
        raise NotImplementedError(
            "Currently not implemented in for local Huggingface!"
        )

    def _context_relevance_endpoint(self, input: str) -> float:
        raise NotImplementedError(
            "Currently not implemented in for local Huggingface!"
        )

    def _positive_sentiment_endpoint(self, input: str) -> float:
        raise NotImplementedError(
            "Currently not implemented in for local Huggingface!"
        )

    def _toxic_endpoint(self, input: str) -> float:
        raise NotImplementedError(
            "Currently not implemented in for local Huggingface!"
        )

    def _summarized_groundedness_endpoint(self, input: str) -> float:
        raise NotImplementedError(
            "Currently not implemented in for local Huggingface!"
        )

    # TODEP
    @_tci
    def _doc_groundedness(self, premise: str, hypothesis: str) -> float:
        tokenizer, model = self._retrieve_tokenizer_and_model(
            HUGS_DOCNLI_MODEL_PATH, tokenizer_kwargs={"use_fast": False}
        )
        with torch.no_grad():
            tokens = tokenizer(
                premise, hypothesis, truncation=False, return_tensors="pt"
            )
            output = model(tokens["input_ids"])
            prediction = torch.softmax(output["logits"][0], -1).tolist()
        return prediction[0]

    def _pii_detection_endpoint(self, input: str) -> List[float]:
        raise NotImplementedError(
            "Currently not implemented in for local Huggingface!"
        )

    def _pii_detection_with_cot_reasons_endpoint(
        self, input: str
    ) -> Tuple[List[float], Dict[str, str]]:
        raise NotImplementedError(
            "Currently not implemented in for local Huggingface!"
        )

    def _hallucination_evaluator_endpoint(self, input: str) -> float:
        raise NotImplementedError(
            "Currently not implemented in for local Huggingface!"
        )
