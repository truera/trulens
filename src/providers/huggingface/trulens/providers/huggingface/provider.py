from abc import abstractmethod
from concurrent.futures import wait
import functools
from inspect import signature
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import requests
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from trulens.core.feedback import Endpoint
from trulens.core.feedback import Provider
from trulens.core.utils.python import Future
from trulens.core.utils.python import locals_except
from trulens.core.utils.threading import ThreadPoolExecutor
from trulens.feedback import prompts
from trulens.feedback.dummy.endpoint import DummyEndpoint
from trulens.providers.huggingface.endpoint import HuggingfaceEndpoint

logger = logging.getLogger(__name__)

# Cannot put these inside Huggingface since it interferes with pydantic.BaseModel.

HUGS_SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HUGS_TOXIC_API_URL = (
    "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"
)
HUGS_LANGUAGE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"
HUGS_NLI_API_URL = "https://api-inference.huggingface.co/models/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
HUGS_DOCNLI_API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
HUGS_DOCNLI_MODEL_PATH = (
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
)
HUGS_PII_DETECTION_API_URL = (
    "https://api-inference.huggingface.co/models/bigcode/starpii"
)
HUGS_CONTEXT_RELEVANCE_API_URL = (
    "https://api-inference.huggingface.co/models/truera/context_relevance"
)
HUGS_HALLUCINATION_API_URL = "https://api-inference.huggingface.co/models/vectara/hallucination_evaluation_model"


# TODO: move this to a more general place and apply it to other feedbacks that need it.
def _tci(func):  # "typecheck inputs"
    """
    Decorate a method to validate its inputs against its signature. Also make
    sure string inputs are non-empty.
    """

    sig = signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bindings = sig.bind(*args, **kwargs)

        for param, annot in sig.parameters.items():
            if param == "self":
                continue
            if annot is not None:
                pident = f"Input `{param}` to `{func.__name__}`"
                v = bindings.arguments[param]

                typ_origin = get_origin(annot.annotation)
                if typ_origin == Union:
                    annotation = get_args(annot.annotation)
                    annotation_name = (
                        "(" + ", ".join(a.__name__ for a in annotation) + ")"
                    )
                elif typ_origin:
                    annotation = typ_origin
                    annotation_name = annotation.__name__
                else:
                    annotation = annot.annotation
                    annotation_name = annot.annotation.__name__

                if not isinstance(v, annotation):
                    raise TypeError(
                        f"{pident} must be of type `{annotation_name}` but was `{type(v).__name__}` instead."
                    )
                if annot.annotation is str:
                    if len(v) == 0:
                        raise ValueError(f"{pident} must be non-empty.")

        return func(*bindings.args, **bindings.kwargs)

    wrapper.__signature__ = sig

    return wrapper


class HuggingfaceBase(Provider):
    """
    Out of the box feedback functions calling Huggingface.
    """

    @abstractmethod
    def _language_scores_endpoint(self, text: str) -> Dict[str, float]: ...

    # TODEP
    @_tci
    @abstractmethod
    def _doc_groundedness(self, premise: str, hypothesis: str) -> float: ...

    @abstractmethod
    def _context_relevance_endpoint(self, input: str) -> float: ...

    @abstractmethod
    def _positive_sentiment_endpoint(self, input: str) -> float: ...

    @abstractmethod
    def _toxic_endpoint(self, input: str) -> float: ...

    @abstractmethod
    def _summarized_groundedness_endpoint(self, input: str) -> float: ...

    @abstractmethod
    def _pii_detection_endpoint(self, input: str) -> List[float]: ...

    @abstractmethod
    def _pii_detection_with_cot_reasons_endpoint(
        self, input: str
    ) -> Tuple[List[float], Dict[str, str]]: ...

    @abstractmethod
    def _hallucination_evaluator_endpoint(self, input: str) -> float: ...

    # TODEP
    @_tci
    def language_match(self, text1: str, text2: str) -> Tuple[float, Dict]:
        """
        Uses Huggingface's papluca/xlm-roberta-base-language-detection model. A
        function that uses language detection on `text1` and `text2` and
        calculates the probit difference on the language detected on text1. The
        function is: `1.0 - (|probit_language_text1(text1) -
        probit_language_text1(text2))`

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.huggingface import Huggingface
            huggingface_provider = Huggingface()

            feedback = Feedback(huggingface_provider.language_match).on_input_output()
            ```

            The `on_input_output()` selector can be changed. See [Feedback Function
            Guide](https://www.trulens.org/trulens/feedback_function_guide/)

        Args:
            text1 (str): Text to evaluate.
            text2 (str): Comparative text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "different languages" and 1 being "same languages".
        """
        with ThreadPoolExecutor(max_workers=2) as tpool:
            max_length = 500
            f_scores1: Future[Dict] = tpool.submit(
                self._language_scores_endpoint, text=text1[:max_length]
            )
            f_scores2: Future[Dict] = tpool.submit(
                self._language_scores_endpoint, text=text2[:max_length]
            )

        wait([f_scores1, f_scores2])

        scores1: Dict = f_scores1.result()
        scores2: Dict = f_scores2.result()

        langs = list(scores1.keys())
        prob1 = np.array([scores1[k] for k in langs])
        prob2 = np.array([scores2[k] for k in langs])
        diff = prob1 - prob2

        l1: float = float(1.0 - (np.linalg.norm(diff, ord=1)) / 2.0)

        return l1, dict(text1_scores=scores1, text2_scores=scores2)

    def groundedness_measure_with_nli(
        self, source: str, statement: str
    ) -> Tuple[float, dict]:
        """
        A measure to track if the source material supports each sentence in the statement using an NLI model.

        First the response will be split into statements using a sentence tokenizer.The NLI model will process each statement using a natural language inference model, and will use the entire source.

        Example:

            ```
            from trulens.core import Feedback
            from trulens.providers.huggingface import Huggingface

            huggingface_provider = Huggingface()

            f_groundedness = (
                Feedback(huggingface_provider.groundedness_measure_with_nli)
                .on(context)
                .on_output()
            ```

        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 (not grounded) and 1.0 (grounded) and a string containing the reasons for the evaluation.
        """
        nltk.download("punkt_tab", quiet=True)
        groundedness_scores = {}

        reasons_str = ""
        if isinstance(source, list):
            source = " ".join(map(str, source))
        hypotheses = sent_tokenize(statement)
        for i, hypothesis in enumerate(hypotheses):
            score = self._doc_groundedness(
                premise=source, hypothesis=hypothesis
            )
            reasons_str = reasons_str + str.format(
                prompts.GROUNDEDNESS_REASON_TEMPLATE,
                statement_sentence=hypothesis,
                supporting_evidence="[Doc NLI Used full source]",
                score=score * 10,
            )
            groundedness_scores[f"statement_{i}"] = score
        average_groundedness_score = float(
            np.mean(list(groundedness_scores.values()))
        )
        return average_groundedness_score, {"reasons": reasons_str}

    @_tci
    def context_relevance(self, prompt: str, context: str) -> float:
        """
        Uses Huggingface's truera/context_relevance model, a
        model that uses computes the relevance of a given context to the prompt.
        The model can be found at https://huggingface.co/truera/context_relevance.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.huggingface import Huggingface
            huggingface_provider = Huggingface()

            feedback = (
                Feedback(huggingface_provider.context_relevance)
                .on_input()
                .on(context)
                .aggregate(np.mean)
                )
            ```

        Args:
            prompt (str): The given prompt.
            context (str): Comparative contextual information.

        Returns:
            float: A value between 0 and 1. 0 being irrelevant and 1 being a relevant context for addressing the prompt.
        """

        if prompt[len(prompt) - 1] != ".":
            prompt += "."
        ctx_relevance_string = prompt + "<eos>" + context
        return self._context_relevance_endpoint(ctx_relevance_string)

    # TODEP
    @_tci
    def positive_sentiment(self, text: str) -> float:
        """
        Uses Huggingface's cardiffnlp/twitter-roberta-base-sentiment model. A
        function that uses a sentiment classifier on `text`.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.huggingface import Huggingface
            huggingface_provider = Huggingface()

            feedback = Feedback(huggingface_provider.positive_sentiment).on_output()
            ```

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 (negative sentiment) and 1 (positive sentiment).
        """
        max_length = 500
        truncated_text = text[:max_length]
        return self._positive_sentiment_endpoint(truncated_text)

    # TODEP
    @_tci
    def toxic(self, text: str) -> float:
        """
        Uses Huggingface's martin-ha/toxic-comment-model model. A function that
        uses a toxic comment classifier on `text`.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.providers.huggingface import Huggingface
            huggingface_provider = Huggingface()

            feedback = Feedback(huggingface_provider.toxic).on_output()
            ```

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 (not toxic) and 1 (toxic).
        """
        assert len(text) > 0, "Input cannot be blank."

        max_length = 500
        truncated_text = text[:max_length]
        return self._toxic_endpoint(truncated_text)

    # TODEP
    @_tci
    def _summarized_groundedness(self, premise: str, hypothesis: str) -> float:
        """A groundedness measure best used for summarized premise against simple hypothesis.
        This Huggingface implementation uses NLI.

        Args:
            premise (str): NLI Premise
            hypothesis (str): NLI Hypothesis

        Returns:
            float: NLI Entailment
        """

        if not "." == premise[len(premise) - 1]:
            premise = premise + "."
        nli_string = premise + " " + hypothesis
        return self._summarized_groundedness_endpoint(nli_string)

    def pii_detection(self, text: str) -> float:
        """
        NER model to detect PII.

        Example:

            ```python
            hugs = Huggingface()

            # Define a pii_detection feedback function using HuggingFace.
            f_pii_detection = Feedback(hugs.pii_detection).on_input()
            ```

            The `on(...)` selector can be changed. See [Feedback Function Guide:
            Selectors](https://www.trulens.org/trulens/feedback_function_guide/#selector-details)

        Args:
            text: A text prompt that may contain a PII.

        Returns:
            float: The likelihood that a PII is contained in the input text.
        """
        # Initialize a list to store scores for "NAME" entities
        likelihood_scores = self._pii_detection_endpoint(text)

        # Calculate the sum of all individual likelihood scores (P(A) + P(B) + ...)
        sum_individual_probabilities = sum(likelihood_scores)

        # Initialize the total likelihood for at least one name
        total_likelihood = sum_individual_probabilities

        # Calculate the product of pairwise likelihood scores (P(A and B), P(A and C), ...)
        for i in range(len(likelihood_scores)):
            for j in range(i + 1, len(likelihood_scores)):
                pairwise_likelihood = (
                    likelihood_scores[i] * likelihood_scores[j]
                )
                total_likelihood -= pairwise_likelihood

        score = 1 - total_likelihood

        return score

    def pii_detection_with_cot_reasons(self, text: str):
        """
        NER model to detect PII, with reasons.

        Example:

            ```python
            hugs = Huggingface()

            # Define a pii_detection feedback function using HuggingFace.
            f_pii_detection = Feedback(hugs.pii_detection).on_input()
            ```

            The `on(...)` selector can be changed. See [Feedback Function Guide
            :
            Selectors](https://www.trulens.org/trulens/feedback_function_guide/#selector-details)

            Args:
                text: A text prompt that may contain a name.

            Returns:
                Tuple[float, str]: A tuple containing a the likelihood that a PII is contained in the input text and a string containing what PII is detected (if any).
        """
        likelihood_scores, reasons = (
            self._pii_detection_with_cot_reasons_endpoint(text)
        )

        # Calculate the sum of all individual likelihood scores (P(A) + P(B) + ...)
        sum_individual_probabilities = sum(likelihood_scores)

        # Initialize the total likelihood for at least one name
        total_likelihood = sum_individual_probabilities

        # Calculate the product of pairwise likelihood scores (P(A and B), P(A and C), ...)
        for i in range(len(likelihood_scores)):
            for j in range(i + 1, len(likelihood_scores)):
                pairwise_likelihood = (
                    likelihood_scores[i] * likelihood_scores[j]
                )
                total_likelihood -= pairwise_likelihood

        score = 1 - total_likelihood

        return score, reasons

    @_tci
    def hallucination_evaluator(
        self, model_output: str, retrieved_text_chunks: str
    ) -> float:
        """
        Evaluates the hallucination score for a combined input of two statements as a float 0<x<1 representing a
        true/false boolean. if the return is greater than 0.5 the statement is evaluated as true. if the return is
        less than 0.5 the statement is evaluated as a hallucination.

        Example:

            ```python
            from trulens.providers.huggingface import Huggingface
            huggingface_provider = Huggingface()

            score = huggingface_provider.hallucination_evaluator("The sky is blue. [SEP] Apples are red , the grass is green.")
            ```

        Args:
            model_output (str): This is what an LLM returns based on the text chunks retrieved during RAG

            retrieved_text_chunks (str): These are the text chunks you have retrieved during RAG

        Returns:
            float: Hallucination score
        """
        combined_input = f"{model_output} [SEP] {retrieved_text_chunks}"
        return self._hallucination_evaluator_endpoint(combined_input)


class Huggingface(HuggingfaceBase):
    """
    Out of the box feedback functions calling Huggingface APIs.
    """

    endpoint: Endpoint

    def __init__(
        self,
        name: str = "huggingface",
        endpoint: Optional[Endpoint] = None,
        **kwargs,
    ):
        # NOTE(piotrm): HACK006: pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Create a Huggingface Provider with out of the box feedback functions.

        Example:

            ```python
            from trulens.providers.huggingface import Huggingface
            huggingface_provider = Huggingface()
            ```
        """

        kwargs["name"] = name

        self_kwargs = dict()

        # TODO: figure out why all of this logic is necessary:
        if endpoint is None:
            self_kwargs["endpoint"] = HuggingfaceEndpoint(**kwargs)
        else:
            if isinstance(endpoint, Endpoint):
                self_kwargs["endpoint"] = endpoint
            else:
                self_kwargs["endpoint"] = HuggingfaceEndpoint(**endpoint)

        self_kwargs["name"] = name or "huggingface"

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _language_scores_endpoint(self, text: str) -> Dict[str, float]:
        payload = {"inputs": text}
        hf_response = self.endpoint.post(
            url=HUGS_LANGUAGE_API_URL, payload=payload, timeout=30
        )
        return {r["label"]: r["score"] for r in hf_response}

    def _context_relevance_endpoint(self, input: str) -> float:
        payload = {"inputs": input}
        hf_response = self.endpoint.post(
            url=HUGS_CONTEXT_RELEVANCE_API_URL, payload=payload
        )
        for label in hf_response:
            if label["label"] == "context_relevance":
                return label["score"]
        raise RuntimeError(
            "'context_relevance' not found in huggingface api response."
        )

    def _positive_sentiment_endpoint(self, input: str) -> float:
        payload = {"inputs": input}
        hf_response = self.endpoint.post(
            url=HUGS_SENTIMENT_API_URL, payload=payload
        )
        for label in hf_response:
            if label["label"] == "LABEL_2":
                return float(label["score"])
        raise RuntimeError("LABEL_2 not found in huggingface api response.")

    def _toxic_endpoint(self, input: str) -> float:
        payload = {"inputs": input}
        hf_response = self.endpoint.post(
            url=HUGS_TOXIC_API_URL, payload=payload
        )
        for label in hf_response:
            if label["label"] == "toxic":
                return label["score"]
        raise RuntimeError("toxic not found in huggingface api response.")

    def _summarized_groundedness_endpoint(self, input: str) -> float:
        payload = {"inputs": input}
        hf_response = self.endpoint.post(url=HUGS_NLI_API_URL, payload=payload)
        for label in hf_response:
            if label["label"] == "entailment":
                return label["score"]
        raise RuntimeError("entailment not found in huggingface api response.")

    # TODEP
    @_tci
    def _doc_groundedness(self, premise: str, hypothesis: str) -> float:
        nli_string = premise + " [SEP] " + hypothesis
        payload = {"inputs": nli_string}
        hf_response = self.endpoint.post(
            url=HUGS_DOCNLI_API_URL, payload=payload
        )
        for label in hf_response:
            if label["label"] == "entailment":
                return label["score"]
        raise ValueError(f"Unrecognized output from {HUGS_DOCNLI_API_URL}!")

    def _pii_detection_endpoint(self, input: str) -> List[float]:
        likelihood_scores = []
        payload = {"inputs": input}
        hf_response = self.endpoint.post(
            url=HUGS_PII_DETECTION_API_URL, payload=payload
        )
        # If the response is a dictionary, convert it to a list. This is for when only one name is identified.
        if isinstance(hf_response, dict):
            hf_response = [hf_response]
        if not isinstance(hf_response, list):
            raise ValueError(
                f"Unexpected response from Huggingface API: {hf_response}"
            )
        # Iterate through the entities and extract scores for "NAME" entities
        for entity in hf_response:
            likelihood_scores.append(entity["score"])
        return likelihood_scores

    def _pii_detection_with_cot_reasons_endpoint(
        self, input: str
    ) -> Tuple[List[float], Dict[str, str]]:
        # Initialize a dictionary to store reasons
        reasons = {}
        # Initialize a list to store scores for "NAME" entities
        likelihood_scores = []
        payload = {"inputs": input}
        try:
            hf_response = self.endpoint.post(
                url=HUGS_PII_DETECTION_API_URL, payload=payload
            )
        # TODO: Make error handling more granular so it's not swallowed.
        except Exception:
            logger.debug("No PII was found")
            hf_response = [
                {
                    "entity_group": "NONE",
                    "score": 0.0,
                    "word": np.nan,
                    "start": np.nan,
                    "end": np.nan,
                }
            ]
        # Convert the response to a list if it's not already a list
        if not isinstance(hf_response, list):
            hf_response = [hf_response]
        # Check if the response is a list
        if not isinstance(hf_response, list):
            raise ValueError(
                "Unexpected response from Huggingface API: response should be a list or a dictionary"
            )
        # Iterate through the entities and extract "word" and "score" for "NAME" entities
        for _, entity in enumerate(hf_response):
            reasons[
                f"{entity.get('entity_group')} detected: {entity['word']}"
            ] = f"PII Likelihood: {entity['score']}"
            likelihood_scores.append(entity["score"])
        return likelihood_scores, reasons

    def _hallucination_evaluator_endpoint(self, input: str) -> float:
        payload = {"inputs": input}
        response = self.endpoint.post(
            url=HUGS_HALLUCINATION_API_URL, payload=payload
        )
        if isinstance(response, list):
            # Assuming the list contains the result, check if the first element has a 'score' key
            if "score" not in response[0]:
                raise RuntimeError(
                    f"Error in API request: {response}, please try again once the endpoint has restarted."
                )
            # Extract the score from the first element
            score = response[0]["score"]
        elif isinstance(
            response, requests.Response
        ):  # Check if it's an HTTP response
            if response.status_code != 200:
                raise RuntimeError(
                    f"Error in API request: {response.text}, please try again once the endpoint has restarted."
                )
            output = response.json()
            score = output[0][0]["score"]
        else:
            # If neither list nor HTTP response, raise an error
            raise RuntimeError(
                "Unexpected response type. Please check the API endpoint."
            )
        return score


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


class Dummy(Huggingface):
    """A version of a Huggingface provider that uses a dummy endpoint and thus
    produces fake results without making any networked calls to huggingface."""

    def __init__(
        self,
        name: str = "dummyhugs",
        error_prob: float = 1 / 100,
        loading_prob: float = 1 / 100,
        freeze_prob: float = 1 / 100,
        overloaded_prob: float = 1 / 100,
        alloc: int = 1024 * 1024,
        rpm: float = 600,
        delay: float = 1.0,
        **kwargs,
    ):
        kwargs["name"] = name or "dummyhugs"
        kwargs["endpoint"] = DummyEndpoint(
            name="dummyendhugspoint", **locals_except("self", "name", "kwargs")
        )

        super().__init__(**kwargs)
