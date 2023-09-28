import logging
from multiprocessing.pool import AsyncResult
from typing import Dict

import numpy as np

from trulens_eval.feedback.provider.base import Provider
from trulens_eval.feedback.provider.endpoint import HuggingfaceEndpoint
from trulens_eval.feedback.provider.endpoint.base import DummyEndpoint
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.utils.threading import TP

logger = logging.getLogger(__name__)

# Cannot put these inside Huggingface since it interferes with pydantic.BaseModel.

HUGS_SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HUGS_TOXIC_API_URL = "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"
HUGS_CHAT_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"
HUGS_LANGUAGE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"
HUGS_NLI_API_URL = "https://api-inference.huggingface.co/models/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
HUGS_DOCNLI_API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"

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

    def __init__(self, name: str = None, endpoint=None, **kwargs):
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
        if endpoint is None:
            self_kwargs['endpoint'] = HuggingfaceEndpoint(**kwargs)
        else:
            self_kwargs['endpoint'] = endpoint

        self_kwargs['name'] = name or "huggingface"

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    @_tci
    def language_match(self, text1: str, text2: str) -> float:
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

        max_length = 500
        scores1: AsyncResult[Dict] = TP().promise(
            get_scores, text=text1[:max_length]
        )
        scores2: AsyncResult[Dict] = TP().promise(
            get_scores, text=text2[:max_length]
        )

        scores1: Dict = scores1.get()
        scores2: Dict = scores2.get()

        langs = list(scores1.keys())
        prob1 = np.array([scores1[k] for k in langs])
        prob2 = np.array([scores2[k] for k in langs])
        diff = prob1 - prob2

        l1 = 1.0 - (np.linalg.norm(diff, ord=1)) / 2.0

        return l1, dict(text1_scores=scores1, text2_scores=scores2)

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
                return label['score']

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


class Dummy(Huggingface):

    def __init__(self, name: str = None, **kwargs):
        kwargs['name'] = name or "dummyhugs"
        kwargs['endpoint'] = DummyEndpoint(name="dummyendhugspoint")

        super().__init__(**kwargs)
