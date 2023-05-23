"""
# Feedback Functions

Initialize feedback function providers:

```python
    hugs = Huggingface()
    openai = OpenAI()
```

Run feedback functions. See examples below on how to create them:

```python
    feedbacks = tru.run_feedback_functions(
        chain=chain,
        record=record,
        feedback_functions=[f_lang_match, f_qs_relevance]
    )
```

## Examples:

Non-toxicity of response:

```python
    f_non_toxic = Feedback(hugs.not_toxic).on_response()
```

Language match feedback function:

```python
    f_lang_match = Feedback(hugs.language_match).on(text1="prompt", text2="response")
```

"""

from datetime import datetime
from inspect import Signature
from inspect import signature
import logging
from multiprocessing.pool import AsyncResult
import re
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import openai
import requests
from tqdm.auto import tqdm

from trulens_eval import feedback_prompts
from trulens_eval.keys import *
from trulens_eval.provider_apis import Endpoint
from trulens_eval.tru_db import JSON, Query, obj_id_of_obj, query_of_path
from trulens_eval.tru_db import Record
from trulens_eval.tru_db import TruDB
from trulens_eval.util import TP

# openai

# (external) feedback-
# provider
# model

# feedback_collator:
# - record, feedback_imp, selector -> dict (input, output, other)

# (external) feedback:
# - *args, **kwargs -> real
# - dict -> real
# - (record, selectors) -> real
# - str, List[str] -> real
#    agg(relevance(str, str[0]),
#        relevance(str, str[1])
#    ...)

# (internal) feedback_imp:
# - Option 1 : input, output -> real
# - Option 2: dict (input, output, other) -> real

Selection = Union[Query, str]
# "prompt" or "input" mean overall chain input text
# "response" or "output"mean overall chain output text
# Otherwise a Query is a path into a record structure.

PROVIDER_CLASS_NAMES = ['OpenAI', 'Huggingface', 'Cohere']


def check_provider(cls_or_name: Union[Type, str]) -> None:
    if isinstance(cls_or_name, str):
        cls_name = cls_or_name
    else:
        cls_name = cls_or_name.__name__

    assert cls_name in PROVIDER_CLASS_NAMES, f"Unsupported provider class {cls_name}"


class Feedback():

    def __init__(
        self,
        imp: Optional[Callable] = None,
        selectors: Optional[Dict[str, Selection]] = None,
        feedback_id: Optional[str] = None
    ):
        """
        A Feedback function container.

        Parameters:
        
        - imp: Optional[Callable] -- implementation of the feedback function.
        - selectors: Optional[Dict[str, Selection]] -- mapping of implementation
          argument names to where to get them from a record.
        """

        # Verify that `imp` expects the arguments specified in `selectors`:
        if imp is not None and selectors is not None:
            sig: Signature = signature(imp)
            for argname in selectors.keys():
                assert argname in sig.parameters, (
                    f"{argname} is not an argument to {imp.__name__}. "
                    f"Its arguments are {list(sig.parameters.keys())}."
                )

        self.imp = imp
        self.selectors = selectors

        if feedback_id is not None:
            self._feedback_id = feedback_id

        if imp is not None and selectors is not None:
            # These are for serialization to/from json and for db storage.

            assert hasattr(
                imp, "__self__"
            ), "Feedback implementation is not a method (it may be a function)."
            self.provider = imp.__self__
            check_provider(self.provider.__class__.__name__)
            self.imp_method_name = imp.__name__
            self._json = self.to_json()
            self._feedback_id = feedback_id or obj_id_of_obj(self._json, prefix="feedback")
            self._json['feedback_id'] = self._feedback_id

    @staticmethod
    def evaluate_deferred(tru: 'Tru'):
        db = tru.db

        def prepare_feedback(row):
            record_json = row.record_json

            feedback = Feedback.of_json(row.feedback_json)
            feedback.run_and_log(record_json=record_json, tru=tru)

        feedbacks = db.get_feedback()

        for i, row in feedbacks.iterrows():
            if row.status == 0:
                tqdm.write(f"Starting run for row {i}.")

                TP().runlater(prepare_feedback, row)
            elif row.status in [1]:
                now = datetime.now().timestamp()
                if now - row.last_ts > 30:
                    tqdm.write(f"Incomplete row {i} last made progress over 30 seconds ago. Retrying.")
                    TP().runlater(prepare_feedback, row)
                else:
                    tqdm.write(f"Incomplete row {i} last made progress less than 30 seconds ago. Giving it more time.")

            elif row.status in [-1]:
                now = datetime.now().timestamp()
                if now - row.last_ts > 60*5:
                    tqdm.write(f"Failed row {i} last made progress over 5 minutes ago. Retrying.")
                    TP().runlater(prepare_feedback, row)
                else:
                    tqdm.write(f"Failed row {i} last made progress less than 5 minutes ago. Not touching it for now.")

            elif row.status == 2:
                pass

        # TP().finish()
        # TP().runrepeatedly(runner)

    @property
    def json(self):
        assert hasattr(self, "_json"), "Cannot json-size partially defined feedback function."
        return self._json

    @property
    def feedback_id(self):
        assert hasattr(self, "_feedback_id"), "Cannot get id of partially defined feedback function."
        return self._feedback_id

    @staticmethod
    def selection_to_json(select: Selection) -> dict:
        if isinstance(select, str):
            return select
        elif isinstance(select, Query):
            return select._path
        else:
            raise ValueError(f"Unknown selection type {type(select)}.")

    @staticmethod
    def selection_of_json(obj: Union[List, str]) -> Selection:
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, (List, Tuple)):
            return query_of_path(obj)  # TODO
        else:
            raise ValueError(f"Unknown selection encoding of type {type(obj)}.")

    def to_json(self) -> dict:
        selectors_json = {
            k: Feedback.selection_to_json(v) for k, v in self.selectors.items()
        }
        return {
            'selectors': selectors_json,
            'imp_method_name': self.imp_method_name,
            'provider': self.provider.to_json()
        }

    @staticmethod
    def of_json(obj) -> 'Feedback':
        assert "selectors" in obj, "Feedback encoding has no 'selectors' field."
        assert "imp_method_name" in obj, "Feedback encoding has no 'imp_method_name' field."
        assert "provider" in obj, "Feedback encoding has no 'provider' field."

        imp_method_name = obj['imp_method_name']
        selectors = {
            k: Feedback.selection_of_json(v)
            for k, v in obj['selectors'].items()
        }
        provider = Provider.of_json(obj['provider'])

        assert hasattr(
            provider, imp_method_name
        ), f"Provider {provider.__name__} has no feedback function {imp_method_name}."
        imp = getattr(provider, imp_method_name)

        return Feedback(imp, selectors=selectors)

    def on_multiple(
        self,
        multiarg: str,
        each_query: Optional[Query] = None,
        agg: Callable = np.mean
    ) -> 'Feedback':
        """
        Create a variant of `self` whose implementation will accept multiple
        values for argument `multiarg`, aggregating feedback results for each.
        Optionally each input element is further projected with `each_query`.

        Parameters:

        - multiarg: str -- implementation argument that expects multiple values.
        - each_query: Optional[Query] -- a query providing the path from each
          input to `multiarg` to some inner value which will be sent to `self.imp`.
        """

        def wrapped_imp(**kwargs):
            assert multiarg in kwargs, f"Feedback function expected {multiarg} keyword argument."

            multi = kwargs[multiarg]

            assert isinstance(
                multi, Sequence
            ), f"Feedback function expected a sequence on {multiarg} argument."

            rets: List[AsyncResult[float]] = []

            for aval in multi:

                if each_query is not None:
                    aval = TruDB.project(query=each_query, obj=aval)

                kwargs[multiarg] = aval

                rets.append(TP().promise(self.imp, **kwargs))

            rets: List[float] = list(map(lambda r: r.get(), rets))

            rets = np.array(rets)

            return agg(rets)

        wrapped_imp.__name__ = self.imp.__name__

        wrapped_imp.__self__ = self.imp.__self__ # needed for serialization

        # Copy over signature from wrapped function. Otherwise signature of the
        # wrapped method will include just kwargs which is insufficient for
        # verify arguments (see Feedback.__init__).
        wrapped_imp.__signature__ = signature(self.imp)

        return Feedback(imp=wrapped_imp, selectors=self.selectors)

    def on_prompt(self, arg: str = "text"):
        """
        Create a variant of `self` that will take in the main chain input or
        "prompt" as input, sending it as an argument `arg` to implementation.
        """

        return Feedback(imp=self.imp, selectors={arg: "prompt"})

    on_input = on_prompt

    def on_response(self, arg: str = "text"):
        """
        Create a variant of `self` that will take in the main chain output or
        "response" as input, sending it as an argument `arg` to implementation.
        """

        return Feedback(imp=self.imp, selectors={arg: "response"})

    on_output = on_response

    def on(self, **selectors):
        """
        Create a variant of `self` with the same implementation but the given `selectors`.
        """

        return Feedback(imp=self.imp, selectors=selectors)

    def run_on_record(self, chain_json: JSON, record_json: JSON) -> Any:
        """
        Run the feedback function on the given `record`. The `chain` that
        produced the record is also required to determine input/output argument
        names.
        """

        if 'record_id' not in record_json:
            record_json['record_id'] = None

        try:
            ins = self.extract_selection(chain_json=chain_json, record_json=record_json)
            ret = self.imp(**ins)
            
            return {
                '_success': True,
                'feedback_id': self.feedback_id,
                'record_id': record_json['record_id'],
                self.name: ret
            }
        
        except Exception as e:
            return {
                '_success': False,
                'feedback_id': self.feedback_id,
                'record_id': record_json['record_id'],
                '_error': str(e)
            }

    def run_and_log(self, record_json: JSON, tru: 'Tru') -> None:
        record_id = record_json['record_id']
        chain_id = record_json['chain_id']
        
        ts_now = datetime.now().timestamp()

        db = tru.db

        try:
            db.insert_feedback(
                record_id=record_id,
                feedback_id=self.feedback_id,
                last_ts = ts_now,
                status = 1 # in progress
            )

            chain_json = db.get_chain(chain_id=chain_id)

            res = self.run_on_record(chain_json=chain_json, record_json=record_json)

        except Exception as e:
            print(e)
            res = {
                '_success': False,
                'feedback_id': self.feedback_id,
                'record_id': record_json['record_id'],
                '_error': str(e)
            }

        ts_now = datetime.now().timestamp()

        if res['_success']:
            db.insert_feedback(
                record_id=record_id,
                feedback_id=self.feedback_id,
                last_ts = ts_now,
                status = 2, # done and good
                result_json=res,
                total_cost=-1.0, # todo
                total_tokens=-1  # todo
            )
        else:
            # TODO: indicate failure better
            db.insert_feedback(
                record_id=record_id,
                feedback_id=self.feedback_id,
                last_ts = ts_now,
                status = -1, # failure
                result_json=res,
                total_cost=-1.0, # todo
                total_tokens=-1  # todo
            )

    @property
    def name(self):
        """
        Name of the feedback function. Presently derived from the name of the
        function implementing it.
        """

        return self.imp.__name__

    def extract_selection(
            self,
            chain_json: Dict,
            record_json: Dict
        ) -> Dict[str, Any]:
        """
        Given the `chain` that produced the given `record`, extract from
        `record` the values that will be sent as arguments to the implementation
        as specified by `self.selectors`.
        """

        ret = {}

        for k, v in self.selectors.items():
            if isinstance(v, Query):
                q = v

            elif v == "prompt" or v == "input":
                if len(chain_json['input_keys']) > 1:
                    #logging.warn(
                    #    f"Chain has more than one input, guessing the first one is prompt."
                    #)
                    pass

                input_key = chain_json['input_keys'][0]

                q = Record.chain._call.args.inputs[input_key]

            elif v == "response" or v == "output":
                if len(chain_json['output_keys']) > 1:
                    #logging.warn(
                    #    "Chain has more than one ouput, guessing the first one is response."
                    #)
                    pass

                output_key = chain_json['output_keys'][0]

                q = Record.chain._call.rets[output_key]

            else:
                raise RuntimeError(f"Unhandled selection type {type(v)}.")

            val = TruDB.project(query=q, record_json=record_json, chain_json=chain_json)
            ret[k] = val

        return ret


pat_1_10 = re.compile(r"\s*([1-9][0-9]*)\s*")


def _re_1_10_rating(str_val):
    matches = pat_1_10.fullmatch(str_val)
    if not matches:
        # Try soft match
        matches = re.search('[1-9][0-9]*', str_val)
        if not matches:
            logging.warn(f"1-10 rating regex failed to match on: '{str_val}'")
            return -10  # so this will be reported as -1 after division by 10

    return int(matches.group())


class Provider():

    @staticmethod
    def of_json(obj: Dict) -> 'Provider':
        cls_name = obj['class']
        check_provider(cls_name)

        cls = eval(cls_name)
        kwargs = {k: v for k, v in obj.items() if k != "class"}

        return cls(**kwargs)

    def to_json(self: 'Provider', **extras) -> Dict:
        obj = {'class': self.__class__.__name__}
        obj.update(**extras)
        return obj

class OpenAI(Provider):

    def __init__(self, model_engine: str = "gpt-3.5-turbo"):
        """
        A set of OpenAI Feedback Functions.

        Parameters:

        - model_engine (str, optional): The specific model version. Defaults to
          "gpt-3.5-turbo".
        """
        self.model_engine = model_engine
        self.endpoint = Endpoint(name="openai")

    def to_json(self) -> Dict:
        return Provider.to_json(self, model_engine=self.model_engine)

    def _moderation(self, text: str):
        return self.endpoint.run_me(
            lambda: openai.Moderation.create(input=text)
        )

    def moderation_not_hate(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is hate
        speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "hate" and 1 being "not
            hate".
        """
        openai_response = self._moderation(text)
        return 1 - float(
            openai_response["results"][0]["category_scores"]["hate"]
        )

    def moderation_not_hatethreatening(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is
        threatening speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "threatening" and 1 being
            "not threatening".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["hate/threatening"]
        )

    def moderation_not_selfharm(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        self harm.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "self harm" and 1 being "not
            self harm".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["self-harm"]
        )

    def moderation_not_sexual(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is sexual
        speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "sexual" and 1 being "not
            sexual".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual"]
        )

    def moderation_not_sexualminors(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        sexual minors.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "sexual minors" and 1 being
            "not sexual minors".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual/minors"]
        )

    def moderation_not_violence(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        violence.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "violence" and 1 being "not
            violence".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence"]
        )

    def moderation_not_violencegraphic(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "graphic violence" and 1
            being "not graphic violence".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence/graphic"]
        )

    def qs_relevance(self, question: str, statement: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the relevance of the statement to the question.

        Parameters:
            question (str): A question being asked. statement (str): A statement
            to the question.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: openai.ChatCompletion.create(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    feedback_prompts.QS_RELEVANCE,
                                    question=question,
                                    statement=statement
                                )
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def relevance(self, prompt: str, response: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the relevance of the response to a prompt.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: openai.ChatCompletion.create(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    feedback_prompts.PR_RELEVANCE,
                                    prompt=prompt,
                                    response=response
                                )
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def model_agreement(self, prompt: str, response: str) -> float:
        """
        Uses OpenAI's Chat GPT Model. A function that gives Chat GPT the same
        prompt and gets a response, encouraging truthfulness. A second template
        is given to Chat GPT with a prompt that the original response is
        correct, and measures whether previous Chat GPT's response is similar.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not in agreement" and 1
            being "in agreement".
        """
        oai_chat_response = OpenAI().endpoint_openai.run_me(
            lambda: openai.ChatCompletion.create(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": feedback_prompts.CORRECT_SYSTEM_PROMPT
                    }, {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )["choices"][0]["message"]["content"]
        )
        agreement_txt = _get_answer_agreement(
            prompt, response, oai_chat_response, self.model_engine
        )
        return _re_1_10_rating(agreement_txt) / 10

    def sentiment(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the sentiment of some text.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """

        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: openai.ChatCompletion.create(
                    model=self.model_engine,
                    temperature=0.5,
                    messages=[
                        {
                            "role": "system",
                            "content": feedback_prompts.SENTIMENT_SYSTEM_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        )


def _get_answer_agreement(prompt, response, check_response, model_engine):
    print("DEBUG")
    print(feedback_prompts.AGREEMENT_SYSTEM_PROMPT % (prompt, response))
    print("MODEL ANSWER")
    print(check_response)
    oai_chat_response = OpenAI().endpoint.run_me(
        lambda: openai.ChatCompletion.create(
            model=model_engine,
            temperature=0.5,
            messages=[
                {
                    "role":
                        "system",
                    "content":
                        feedback_prompts.AGREEMENT_SYSTEM_PROMPT %
                        (prompt, response)
                }, {
                    "role": "user",
                    "content": check_response
                }
            ]
        )["choices"][0]["message"]["content"]
    )
    return oai_chat_response


class Huggingface(Provider):

    SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    TOXIC_API_URL = "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"
    CHAT_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"
    LANGUAGE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"

    def __init__(self):
        """A set of Huggingface Feedback Functions. Utilizes huggingface api-inference
        """
        self.endpoint = Endpoint(
            name="huggingface", post_headers=get_huggingface_headers()
        )

    def language_match(self, text1: str, text2: str) -> float:
        """
        Uses Huggingface's papluca/xlm-roberta-base-language-detection model. A
        function that uses language detection on `text1` and `text2` and
        calculates the probit difference on the language detected on text1. The
        function is: `1.0 - (|probit_language_text1(text1) -
        probit_language_text1(text2))`
        
        Parameters:
        
            text1 (str): Text to evaluate.

            text2 (str): Comparative text to evaluate.

        Returns:

            float: A value between 0 and 1. 0 being "different languages" and 1
            being "same languages".
        """

        def get_scores(text):
            payload = {"inputs": text}
            hf_response = self.endpoint.post(
                url=Huggingface.LANGUAGE_API_URL, payload=payload, timeout=30
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

        return l1

    def positive_sentiment(self, text: str) -> float:
        """
        Uses Huggingface's cardiffnlp/twitter-roberta-base-sentiment model. A
        function that uses a sentiment classifier on `text`.
        
        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """
        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}

        hf_response = self.endpoint.post(
            url=Huggingface.SENTIMENT_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'LABEL_2':
                return label['score']

    def not_toxic(self, text: str) -> float:
        """
        Uses Huggingface's martin-ha/toxic-comment-model model. A function that
        uses a toxic comment classifier on `text`.
        
        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "toxic" and 1 being "not
            toxic".
        """
        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}
        hf_response = self.endpoint.post(
            url=Huggingface.TOXIC_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'toxic':
                return label['score']


# cohere
class Cohere(Provider):

    def __init__(self, model_engine='large'):
        Cohere().endpoint = Endpoint(name="cohere")
        self.model_engine = model_engine

    def to_json(self) -> Dict:
        return Provider.to_json(self, model_engine=self.model_engine)

    def sentiment(
        self,
        text,
    ):
        return int(
            Cohere().endpoint.run_me(
                lambda: get_cohere_agent().classify(
                    model=self.model_engine,
                    inputs=[text],
                    examples=feedback_prompts.COHERE_SENTIMENT_EXAMPLES
                )[0].prediction
            )
        )

    def not_disinformation(self, text):
        return int(
            Cohere().endpoint.run_me(
                lambda: get_cohere_agent().classify(
                    model=self.model_engine,
                    inputs=[text],
                    examples=feedback_prompts.COHERE_NOT_DISINFORMATION_EXAMPLES
                )[0].prediction
            )
        )
