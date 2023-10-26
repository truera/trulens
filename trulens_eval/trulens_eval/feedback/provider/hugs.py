from concurrent.futures import Future
from concurrent.futures import wait
import logging
from multiprocessing.pool import AsyncResult
from typing import Dict, Optional, Tuple

import numpy as np

from trulens_eval.feedback.provider.base import Provider
from trulens_eval.feedback.provider.endpoint import HuggingfaceEndpoint
from trulens_eval.feedback.provider.endpoint.base import DummyEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.python import locals_except
from trulens_eval.utils.threading import ThreadPoolExecutor
from trulens_eval.utils.threading import TP

logger = logging.getLogger(__name__)

# Cannot put these inside Huggingface since it interferes with pydantic.BaseModel.

HUGS_SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HUGS_TOXIC_API_URL = "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"
HUGS_CHAT_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"
HUGS_LANGUAGE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"
HUGS_NLI_API_URL = "https://api-inference.huggingface.co/models/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
HUGS_DOCNLI_API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
HUGS_PII_DETECTION_API_URL = "https://api-inference.huggingface.co/models/bigcode/starpii"

import functools
from inspect import signature


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
                if not isinstance(v, annot.annotation):
                    raise TypeError(
                        f"{pident} must be of type `{annot.annotation.__name__}` but was `{type(v).__name__}` instead."
                    )
                if annot.annotation is str:
                    if len(v) == 0:
                        raise ValueError(f"{pident} must be non-empty.")

        return func(*bindings.args, **bindings.kwargs)

    wrapper.__signature__ = sig

    return wrapper


class Huggingface(Provider):
    """
    Out of the box feedback functions calling Huggingface APIs.
    """

    endpoint: Endpoint

    def __init__(self, name: Optional[str] = None, endpoint=None, **kwargs):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Create a Huggingface Provider with out of the box feedback functions.

        **Usage:**
        ```python
        from trulens_eval.feedback.provider.hugs import Huggingface
        huggingface_provider = Huggingface()
        ```

        Args:
            endpoint (Endpoint): Internal Usage for DB serialization
        """

        kwargs['name'] = name

        self_kwargs = dict()

        # TODO: figure out why all of this logic is necessary:
        if endpoint is None:
            self_kwargs['endpoint'] = HuggingfaceEndpoint(**kwargs)
        else:
            if isinstance(endpoint, Endpoint):
                self_kwargs['endpoint'] = endpoint
            else:
                self_kwargs['endpoint'] = HuggingfaceEndpoint(**endpoint)

        self_kwargs['name'] = name or "huggingface"

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    # TODEP
    @_tci
    def language_match(self, text1: str, text2: str) -> Tuple[float, Dict]:
        """
        Uses Huggingface's papluca/xlm-roberta-base-language-detection model. A
        function that uses language detection on `text1` and `text2` and
        calculates the probit difference on the language detected on text1. The
        function is: `1.0 - (|probit_language_text1(text1) -
        probit_language_text1(text2))`
        
        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.hugs import Huggingface
        huggingface_provider = Huggingface()

        feedback = Feedback(huggingface_provider.language_match).on_input_output() 
        ```
        The `on_input_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text1 (str): Text to evaluate.
            text2 (str): Comparative text to evaluate.

        Returns:

            float: A value between 0 and 1. 0 being "different languages" and 1
            being "same languages".
        """

        def get_scores(text):
            payload = {"inputs": text}
            hf_response = self.endpoint.post(
                url=HUGS_LANGUAGE_API_URL, payload=payload, timeout=30
            )
            return {r['label']: r['score'] for r in hf_response}

        with ThreadPoolExecutor(max_workers=2) as tpool:
            max_length = 500
            f_scores1: Future[Dict] = tpool.submit(
                get_scores, text=text1[:max_length]
            )
            f_scores2: Future[Dict] = tpool.submit(
                get_scores, text=text2[:max_length]
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

    # TODEP
    @_tci
    def positive_sentiment(self, text: str) -> float:
        """
        Uses Huggingface's cardiffnlp/twitter-roberta-base-sentiment model. A
        function that uses a sentiment classifier on `text`.
        
        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.hugs import Huggingface
        huggingface_provider = Huggingface()

        feedback = Feedback(huggingface_provider.positive_sentiment).on_output() 
        ```
        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """

        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}

        hf_response = self.endpoint.post(
            url=HUGS_SENTIMENT_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'LABEL_2':
                return float(label['score'])

        raise RuntimeError("LABEL_2 not found in huggingface api response.")

    # TODEP
    @_tci
    def not_toxic(self, text: str) -> float:
        """
        Uses Huggingface's martin-ha/toxic-comment-model model. A function that
        uses a toxic comment classifier on `text`.
        
        **Usage:**
        ```python
        from trulens_eval import Feedback
        from trulens_eval.feedback.provider.hugs import Huggingface
        huggingface_provider = Huggingface()

        feedback = Feedback(huggingface_provider.not_toxic).on_output() 
        ```
        The `on_output()` selector can be changed. See [Feedback Function
        Guide](https://www.trulens.org/trulens_eval/feedback_function_guide/)

        
        Args:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "toxic" and 1 being "not
            toxic".
        """

        assert len(text) > 0, "Input cannot be blank."

        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}
        hf_response = self.endpoint.post(
            url=HUGS_TOXIC_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'toxic':
                return label['score']

        raise RuntimeError("LABEL_2 not found in huggingface api response.")

    # TODEP
    @_tci
    def _summarized_groundedness(self, premise: str, hypothesis: str) -> float:
        """ A groundedness measure best used for summarized premise against simple hypothesis.
        This Huggingface implementation uses NLI.

        Args:
            premise (str): NLI Premise
            hypothesis (str): NLI Hypothesis

        Returns:
            float: NLI Entailment
        """

        if not '.' == premise[len(premise) - 1]:
            premise = premise + '.'
        nli_string = premise + ' ' + hypothesis
        payload = {"inputs": nli_string}
        hf_response = self.endpoint.post(url=HUGS_NLI_API_URL, payload=payload)

        for label in hf_response:
            if label['label'] == 'entailment':
                return label['score']

        raise RuntimeError("LABEL_2 not found in huggingface api response.")

    # TODEP
    @_tci
    def _doc_groundedness(self, premise: str, hypothesis: str) -> float:
        """
        A groundedness measure for full document premise against hypothesis.
        This Huggingface implementation uses DocNLI. The Hypoethsis still only
        works on single small hypothesis.

        Args:
            premise (str): NLI Premise
            hypothesis (str): NLI Hypothesis

        Returns:
            float: NLI Entailment
        """
        nli_string = premise + ' [SEP] ' + hypothesis
        payload = {"inputs": nli_string}
        hf_response = self.endpoint.post(
            url=HUGS_DOCNLI_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'entailment':
                return label['score']

    def pii_detection(self, text: str) -> float:
        """
        NER model to detect PII.
        **Usage:**
        ```
        hugs = Huggingface()

        # Define a pii_detection feedback function using HuggingFace.
        f_pii_detection = Feedback(hugs.pii_detection).on_input()
        ```
        The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

        Args:
            input (str): A text prompt that may contain a name.

        Returns:
            - float: the likelihood that a name is contained in the input text.
        """

        # Initialize a list to store scores for "NAME" entities
        likelihood_scores = []

        payload = {"inputs": text}

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

        # Calculate the sum of all individual likelihood scores (P(A) + P(B) + ...)
        sum_individual_probabilities = sum(likelihood_scores)

        # Initialize the total likelihood for at least one name
        total_likelihood = sum_individual_probabilities

        # Calculate the product of pairwise likelihood scores (P(A and B), P(A and C), ...)
        for i in range(len(likelihood_scores)):
            for j in range(i + 1, len(likelihood_scores)):
                pairwise_likelihood = likelihood_scores[i] * likelihood_scores[j]
                total_likelihood -= pairwise_likelihood

        score = 1 - total_likelihood

        return score

    def pii_detection_with_cot_reasons(self, text: str):
        """
        NER model to detect PII, with reasons.

        **Usage:**
        ```
        hugs = Huggingface()

        # Define a pii_detection feedback function using HuggingFace.
        f_pii_detection = Feedback(hugs.pii_detection).on_input()
        ```
        The `on(...)` selector can be changed. See [Feedback Function Guide : Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)
        """

        # Initialize a dictionary to store reasons
        reasons = {}

        # Initialize a list to store scores for "NAME" entities
        likelihood_scores = []

        payload = {"inputs": text}

        try:
            hf_response = self.endpoint.post(
                url=HUGS_PII_DETECTION_API_URL, payload=payload
            )

        # TODO: Make error handling more granular so it's not swallowed.
        except Exception as e:
            logger.debug("No PII was found")
            hf_response = [
                {
                    "entity_group": "NONE",
                    "score": 0.0,
                    "word": np.nan,
                    "start": np.nan,
                    "end": np.nan
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
        for i, entity in enumerate(hf_response):
            reasons[f"{entity.get('entity_group')} detected: {entity['word']}"
                   ] = f"PII Likelihood: {entity['score']}"
            likelihood_scores.append(entity["score"])

        # Calculate the sum of all individual likelihood scores (P(A) + P(B) + ...)
        sum_individual_probabilities = sum(likelihood_scores)

        # Initialize the total likelihood for at least one name
        total_likelihood = sum_individual_probabilities

        # Calculate the product of pairwise likelihood scores (P(A and B), P(A and C), ...)
        for i in range(len(likelihood_scores)):
            for j in range(i + 1, len(likelihood_scores)):
                pairwise_likelihood = likelihood_scores[i] * likelihood_scores[j]
                total_likelihood -= pairwise_likelihood

        score = 1 - total_likelihood

        return score, reasons


class Dummy(Huggingface):

    def __init__(
        self,
        name: Optional[str] = None,
        error_prob: float = 1 / 100,
        loading_prob: float = 1 / 100,
        freeze_prob: float = 1 / 100,
        overloaded_prob: float = 1 / 100,
        rpm: float = 600,
        **kwargs
    ):
        kwargs['name'] = name or "dummyhugs"
        kwargs['endpoint'] = DummyEndpoint(
            name="dummyendhugspoint", **locals_except("self", "name", "kwargs")
        )

        super().__init__(**kwargs)
