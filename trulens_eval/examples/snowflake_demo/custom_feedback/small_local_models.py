import os

from scipy.special import expit
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from trulens_eval import Provider

CONTEXT_RELEVANCE_MODEL_PATH = os.getenv(
    "SMALL_LOCAL_MODELS_CONTEXT_RELEVANCE_MODEL_PATH",
    "/trulens_demo/small_local_models/context_relevance",
)


class SmallLocalModels(Provider):

    context_relevance_tokenizer: PreTrainedTokenizerBase = (
        AutoTokenizer.from_pretrained(CONTEXT_RELEVANCE_MODEL_PATH)
    )
    context_relevance_model: PreTrainedModel = (
        AutoModelForSequenceClassification.
        from_pretrained(CONTEXT_RELEVANCE_MODEL_PATH)
    )

    def context_relevance(
        self, question: str, context: str, temperature: float = 0.0
    ) -> float:
        tokenizer = self.context_relevance_tokenizer
        model = self.context_relevance_model
        with torch.no_grad():
            logit = model.forward(
                torch.tensor(
                    tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(f"{question} [SEP] {context}")
                    )
                ).reshape(1, -1)
            ).logits.numpy()
            if logit.size != 1:
                raise ValueError("Unexpected number of results from model!")
            logit = float(logit[0, 0])
            return expit(logit)
