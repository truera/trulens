from scipy.special import expit
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from trulens_eval import Provider


class SmallLocalModels(Provider):

    def __init__(self, context_relevance_model_path: str):
        self.model_config["context_relevance_tokenizer"] = (
            AutoTokenizer.from_pretrained(context_relevance_model_path)
        )
        self.model_config["context_relevance_model"] = (
            AutoModelForSequenceClassification.
            from_pretrained(context_relevance_model_path)
        )

    def context_relevance(
        self, question: str, context: str, temperature: float = 0.0
    ) -> float:
        tokenizer = self.model_config["context_relevance_tokenizer"]
        model = self.model_config["context_relevance_model"]
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
