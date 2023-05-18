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

Question/Statement relevance that is evaluated on a sub-chain input which contains more than one piece of text:

```python
    f_qs_relevance = Feedback(openai.qs_relevance) \
        .on(
            question="input",
            statement=Record.chain.combine_docs_chain._call.args.inputs.input_documents
        ) \
        .on_multiple(multiarg="statement", each_query=Record.page_content)
```

"""

import re
from typing import Any, Callable, Dict, Optional, Sequence, Union

import cohere

import numpy as np
import openai
import requests

from trulens_evalchain import feedback_prompts
from trulens_evalchain import tru
from trulens_evalchain.keys import *
from trulens_evalchain.tru_chain import TruChain
from trulens_evalchain.tru_db import Record
from trulens_evalchain.tru_db import TruDB
from trulens_evalchain.tru_db import Query

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

class Feedback():
    def __init__(self, imp: Optional[Callable] = None, selectors: Optional[Dict[str, Selection]] = None):
        """
        A Feedback function container.

        Args:
        
        - imp: Optional[Callable] -- implementation of the feedback function.
        - selectors: Optional[Dict[str, Selection]] -- mapping of implementation
          argument names to where to get them from a record.
        """

        self.imp = imp
        self.selectors = selectors

    def on_multiple(self, multiarg: str, each_query: Optional[Query] = None, agg: Callable = np.mean) -> 'Feedback':
        """
        Create a variant of `self` whose implementation will accept multiple
        values for argument `multiarg`, aggregating feedback results for each.
        Optionally each input element is further projected with `each_query`.

        Args:

        - multiarg: str -- implementation argument that expects multiple values.
        - each_query: Optional[Query] -- a query providing the path from each
          input to `multiarg` to some inner value which will be sent to `self.imp`.
        """

        def wrapped_imp(**kwargs):
            assert multiarg in kwargs, f"Feedback function expected {multiarg} keyword argument."

            multi = kwargs[multiarg]

            assert isinstance(multi, Sequence), f"Feedback function expected a sequence on {multiarg} argument."

            rets = []

            # TODO: parallelize

            for aval in multi:
                # print(f"multiarg {multiarg} = {aval}")

                if each_query is not None:
                    aval = TruDB.project(query=each_query, obj=aval)
            
                # print(f"multiarg {multiarg} = {aval}")

                kwargs[multiarg] = aval

                ret = self.imp(**kwargs)

                rets.append(ret)

            rets = np.array(rets)

            return agg(rets)

        wrapped_imp.__name__ = self.imp.__name__

        return Feedback(imp=wrapped_imp, selectors=self.selectors)

    def on_prompt(self, arg: str = "text"):
        """
        Create a variant of `self` that will take in the main chain input or
        "prompt" as input, sending it as an argument `arg` to implementation.
        """

        return Feedback(imp=self.imp, selectors={arg:"prompt"})
    
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

    def run(self, chain: TruChain, record: Dict) -> Any:
        """
        Run the feedback function on the given `record`. The `chain` that
        produced the record is also required to determine input/output argument
        names.
        """

        ins = self.extract_selection(chain=chain, record=record)

        print(f"Will run {self.imp} on {ins}.")

        ret = self.imp(**ins)

        return ret

    @property
    def name(self):
        """
        Name of the feedback function. Presently derived from the name of the
        function implementing it.
        """

        return self.imp.__name__

    def extract_selection(self, chain: TruChain, record: dict) -> Dict[str, Any]:
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
                if len(chain.input_keys) > 1:
                    print("WARNING: chain has more than one input, guessing the first one is prompt.")
                input_key = chain.input_keys[0]

                q = Record.chain._call.args.inputs[input_key]

            elif v == "response" or v == "output":
                if len(chain.output_keys) > 1:
                    print("WARNING: chain has more than one ouput, guessing the first one is response.")
                output_key = chain.output_keys[0]

                q = Record.chain._call.rets[output_key]

            else:
                raise RuntimeError(f"Unhandled selection type {type(v)}.")
            
            print(f"q={q._path}")

            val = TruDB.project(query=q, obj=record)
            ret[k] = val

        return ret

def _re_1_10_rating(str_val):
    matches = re.search(re.compile(r"([1-10]+)"), str_val)
    if not matches:
        print(f"WARNING: 1-10 rating regex failed to match on: '{str_val}'")
        return -10 # so this will be reported as -1 after division by 10
    
    return int(matches.group())

class OpenAI():
    def __init__(self, model_engine: str =  "gpt-3.5-turbo"):
        self.model_engine = model_engine

    def _moderation(self, text: str):
        return tru.endpoint_openai.run_me(lambda: openai.Moderation.create(input=text))

    def moderation_not_hate(self, text: str):
        openai_response = self._moderation(text)
        return 1 - float(openai_response["results"][0]["category_scores"]["hate"])
    
    def moderation_not_hatethreatening(self, text: str):
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["hate/threatening"]
        )

    def moderation_not_selfharm(self, text: str):
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["self-harm"]
        )

    def moderation_not_sexual(self, text: str):
        openai_response = self._moderation(text)

        return 1 - int(openai_response["results"][0]["category_scores"]["sexual"])

    def moderation_not_sexualminors(self, text: str):
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual/minors"]
        )

    def moderation_not_violence(self, text: str):
        openai_response = self._moderation(text)

        return 1 - int(openai_response["results"][0]["category_scores"]["violence"])

    def moderation_not_violencegraphic(self, text: str):
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence/graphic"]
        )

    def qs_relevance(self, question: str, statement: str):
        return _re_1_10_rating(
            tru.endpoint_openai.run_me(lambda: openai.ChatCompletion.create(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": str.format(feedback_prompts.QS_RELEVANCE, question=question, statement=statement)
                    }
                ]
            )["choices"][0]["message"]["content"])
        ) / 10


    def relevance(self, prompt: str, response: str):
        return _re_1_10_rating(
            tru.endpoint_openai.run_me(lambda: openai.ChatCompletion.create(
                model=self.model_engine,
                temperature=0.5,
                messages=[
                    {
                        "role": "system",
                        "content": feedback_prompts.RELEVANCE_SYSTEM_PROMPT + prompt
                    }, {
                        "role":
                            "user",
                        "content":
                            feedback_prompts.RELEVANCE_CONTENT_PROMPT + response
                    }
                ]
            )["choices"][0]["message"]["content"])
        ) / 10


    def sentiment(
        self, text: str
    ):
    
        return _re_1_10_rating(
            tru.endpoint_openai.run_me(lambda: openai.ChatCompletion.create(
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
        ))

# outdated interfaces
def openai_moderation_not_hate(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response

    openai_response = tru.endpoint_openai.run_me(lambda: openai.Moderation.create(input=input))

    return 1 - float(openai_response["results"][0]["category_scores"]["hate"])


def openai_moderation_not_hatethreatening(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response

    openai_response = tru.endpoint_openai.run_me(lambda: openai.Moderation.create(input=input))

    return 1 - int(
        openai_response["results"][0]["category_scores"]["hate/threatening"]
    )


def openai_moderation_not_selfharm(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = tru.endpoint_openai.run_me(lambda: openai.Moderation.create(input=input))
    return 1 - int(
        openai_response["results"][0]["category_scores"]["self-harm"]
    )


def openai_moderation_not_sexual(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = tru.endpoint_openai.run_me(lambda: openai.Moderation.create(input=input))
    return 1 - int(openai_response["results"][0]["category_scores"]["sexual"])


def openai_moderation_not_sexualminors(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = tru.endpoint_openai.run_me(lambda: openai.Moderation.create(input=input))
    return 1 - int(
        openai_response["results"][0]["category_scores"]["sexual/minors"]
    )


def openai_moderation_not_violence(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = tru.endpoint_openai.run_me(lambda: openai.Moderation.create(input=input))
    return 1 - int(openai_response["results"][0]["category_scores"]["violence"])


def openai_moderation_not_violencegraphic(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = tru.endpoint_openai.run_me(lambda: openai.Moderation.create(input=input))
    return 1 - int(
        openai_response["results"][0]["category_scores"]["violence/graphic"]
    )


def openai_qs_relevance(question: str, statement: str, model_engine):

    return _re_1_10_rating(
        tru.endpoint_openai.run_me(lambda: openai.ChatCompletion.create(
            model=model_engine,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": str.format(feedback_prompts.QS_RELEVANCE, question=question, statement=statement)
                }
            ]
        )["choices"][0]["message"]["content"])
    ) / 10


def openai_relevance(prompt, response, model_engine):

    return _re_1_10_rating(
        tru.endpoint_openai.run_me(lambda: openai.ChatCompletion.create(
            model=model_engine,
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": feedback_prompts.RELEVANCE_SYSTEM_PROMPT + prompt
                }, {
                    "role":
                        "user",
                    "content":
                        feedback_prompts.RELEVANCE_CONTENT_PROMPT + response
                }
            ]
        )["choices"][0]["message"]["content"])
    ) / 10


def openai_sentiment_function(
    prompt, response, evaluation_choice, model_engine
):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response

    return _re_1_10_rating(
                tru.endpoint_openai.run_me(lambda: openai.ChatCompletion.create(
                    model=model_engine,
                    temperature=0.5,
                    messages=[
                        {
                            "role": "system",
                            "content": feedback_prompts.SENTIMENT_SYSTEM_PROMPT
                        }, {
                            "role": "user",
                            "content": input
                        }
                    ]
                )["choices"][0]["message"]["content"]
            ))
            


# huggingface


class Huggingface():
    SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    TOXIC_API_URL = "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"
    CHAT_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"
    LANGUAGE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"

    def __init__(self):
        pass

    def language_match(self, text1: str, text2: str) -> float:
        # TODO: parallelize

        max_length = 500
        truncated_text = text1[:max_length]
        payload = {"inputs": truncated_text}
        hf_response = tru.endpoint_huggingface.post(url=Huggingface.LANGUAGE_API_URL, payload=payload)
        scores1 = {r['label']: r['score'] for r in hf_response}

        truncated_text = text2[:max_length]
        payload = {"inputs": truncated_text}
        hf_response = tru.endpoint_huggingface.post(url=Huggingface.LANGUAGE_API_URL, payload=payload)
        scores2 = {r['label']: r['score'] for r in hf_response}

        langs = list(scores1.keys())
        prob1 = np.array([scores1[k] for k in langs])
        prob2 = np.array([scores2[k] for k in langs])
        diff = prob1 - prob2
        
        l1 = 1.0 - (np.linalg.norm(diff, ord=1)) / 2.0 

        return l1

    def positive_sentiment(self, text: str):
        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}

        hf_response = tru.endpoint_huggingface.post(url=Huggingface.SENTIMENT_API_URL, payload=payload)

        for label in hf_response:
            if label['label'] == 'LABEL_2':
                return label['score']

    def not_toxic(self, text: str):
        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}
        hf_response = tru.endpoint_huggingface.post(url=Huggingface.TOXIC_API_URL, payload=payload)
        
        for label in hf_response:
            if label['label'] == 'toxic':
                return label['score']


# old interface
def huggingface_language_match(prompt, response, evaluation_choice=None) -> float:
    max_length = 500
    truncated_text = prompt[:max_length]
    payload = {"inputs": truncated_text}

    hf_response = tru.endpoint_huggingface.run_me(lambda: requests.post(
        Huggingface.LANGUAGE_API_URL, headers=HUGGINGFACE_HEADERS, json=payload
    ).json())

    if not (isinstance(hf_response, list) and len(hf_response) > 0):
        raise RuntimeError(hf_response)
    
    hf_response = hf_response[0]


    scores1 = {r['label']: r['score'] for r in hf_response}

    truncated_text = response[:max_length]
    payload = {"inputs": truncated_text}
    hf_response = tru.endpoint_huggingface.run_me(lambda: requests.post(
        Huggingface.LANGUAGE_API_URL, headers=HUGGINGFACE_HEADERS, json=payload
    ).json())

    if not (isinstance(hf_response, list) and len(hf_response) > 0):
        raise RuntimeError(hf_response)
    hf_response = hf_response[0]

    scores2 = {r['label']: r['score'] for r in hf_response}

    langs = list(scores1.keys())
    prob1 = np.array([scores1[k] for k in langs])
    prob2 = np.array([scores2[k] for k in langs])
    diff = prob1 - prob2
    
    l1 = 1.0 - (np.linalg.norm(diff, ord=1)) / 2.0 

    return l1

def huggingface_positive_sentiment(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    max_length = 500
    truncated_text = input[:max_length]
    payload = {"inputs": truncated_text}
    hf_response = tru.endpoint_huggingface.run_me(lambda: requests.post(
        Huggingface.SENTIMENT_API_URL, headers=HUGGINGFACE_HEADERS, json=payload
    ).json()[0])

    for label in hf_response:
        if label['label'] == 'LABEL_2':
            return label['score']


def huggingface_not_toxic(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    max_length = 500
    truncated_text = input[:max_length]
    payload = {"inputs": truncated_text}
    hf_response = tru.endpoint_huggingface.run_me(lambda: requests.post(
        Huggingface.TOXIC_API_URL, headers=HUGGINGFACE_HEADERS, json=payload
    ).json()[0])
    for label in hf_response:
        if label['label'] == 'toxic':
            return label['score']


# cohere

cohere.api_key = config['COHERE_API_KEY']

co = cohere.Client(cohere.api_key)


def cohere_sentiment(prompt, response, evaluation_choice, model_engine):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    return int(
        tru.endpoint_cohere.run_me(lambda: co.classify(
            model=model_engine,
            inputs=[input],
            examples=feedback_prompts.COHERE_SENTIMENT_EXAMPLES
        )[0].prediction)
    )


def cohere_not_disinformation(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    return int(
        tru.endpoint_cohere.run_me(lambda: co.classify(
            model='large',
            inputs=[input],
            examples=feedback_prompts.COHERE_NOT_DISINFORMATION_EXAMPLES
        )[0].prediction)
    )


def _get_answer_agreement(prompt, response, check_response, model_engine):
    oai_chat_response = tru.endpoint_openai.run_me(lambda: openai.ChatCompletion.create(
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
    )["choices"][0]["message"]["content"])
    return oai_chat_response


def openai_factagreement(prompt, response, model_engine):
    if not prompt or prompt == "":
        prompt = f"Finish this thought: {response[:int(len(response)*4.0/5.0)]}"
    payload = {"text": prompt}
    hf_response = requests.post(
        Huggingface.CHAT_API_URL, headers=HUGGINGFACE_HEADERS, json=payload
    ).json()['generated_text']

    # Attempt an honest bot
    oai_chat_response = tru.endpoint_openai.run_me(lambda: openai.ChatCompletion.create(
        model=model_engine,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": feedback_prompts.CORRECT_SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": prompt
            }
        ]
    )["choices"][0]["message"]["content"])

    oai_similarity_response_1 = _get_answer_agreement(
        prompt, response, hf_response, model_engine
    )
    oai_similarity_response_2 = _get_answer_agreement(
        prompt, response, oai_chat_response, model_engine
    )

    #print(f"Prompt: {prompt}\n\nResponse: {response}\n\nHFResp: {hf_response}\n\nOAIResp: {oai_chat_response}\n\nAgree1: {oai_similarity_response_1}\n\nAgree2: {oai_similarity_response_2}\n\n")
    return (
        (
            _re_1_10_rating(oai_similarity_response_1) +
            _re_1_10_rating(oai_similarity_response_2)
        ) / 2
    ) / 10


def get_language_match_function(provider='huggingface', model_engine=None, evaluation_choice=None):
    if provider == "huggingface":

        def huggingface_language_match_feedback_function(prompt, response):
            feedback_function = huggingface_language_match(
                prompt, response, evaluation_choice
            )
            return feedback_function

        return huggingface_language_match_feedback_function

    else:
        raise NotImplementedError(
            "Invalid provider specified. Language match feedback function is only supported for `provider` as `huggingface`"
        )

def get_sentimentpositive_function(provider, model_engine, evaluation_choice):
    if provider == "openai":

        def openai_sentimentpositive_feedback_function(prompt, response):
            feedback_function = openai_sentiment_function(
                prompt, response, evaluation_choice, model_engine
            )
            return feedback_function

        return openai_sentimentpositive_feedback_function

    elif provider == "huggingface":

        def huggingface_sentimentpositive_feedback_function(prompt, response):
            feedback_function = huggingface_positive_sentiment(
                prompt, response, evaluation_choice
            )
            return feedback_function

        return huggingface_sentimentpositive_feedback_function

    elif provider == "cohere":

        def cohere_sentimentpositive_feedback_function(prompt, response):
            feedback_function = cohere_sentiment(
                prompt, response, evaluation_choice, model_engine
            )
            return feedback_function

        return cohere_sentimentpositive_feedback_function

    else:
        raise NotImplementedError(
            "Invalid provider specified. sentiment feedback function is only supported for `provider` as `openai`, `huggingface`, or `cohere`"
        )


def get_qs_relevance_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "relevance feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_qs_relevance_function(prompt, response):
        return openai_qs_relevance(question=prompt, statement=response, model_engine=model_engine)

    return openai_qs_relevance_function


def get_relevance_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "relevance feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_relevance_function(prompt, response):
        return openai_relevance(prompt, response, model_engine)

    return openai_relevance_function


def get_not_hate_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "hate feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_not_hate_function(prompt, response):
        return openai_moderation_not_hate(prompt, response, evaluation_choice)

    return openai_not_hate_function


def get_not_hatethreatening_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "hatethreatening feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_not_hatethreatening_function(prompt, response):
        return openai_moderation_not_hatethreatening(
            prompt, response, evaluation_choice
        )

    return openai_not_hatethreatening_function


def get_not_selfharm_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "selfharm feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_not_selfharm_function(prompt, response):
        return openai_moderation_not_selfharm(
            prompt, response, evaluation_choice
        )

    return openai_not_selfharm_function


def get_not_sexual_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "sexual feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_not_sexual_function(prompt, response):
        return openai_moderation_not_sexual(prompt, response, evaluation_choice)

    return openai_not_sexual_function


def get_not_sexualminors_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "sexualminors feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_not_sexualminors_function(prompt, response):
        return openai_moderation_not_sexual(prompt, response, evaluation_choice)

    return openai_not_sexualminors_function


def get_violence_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "violence feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_not_violence_function(prompt, response):
        return openai_moderation_not_violence(
            prompt, response, evaluation_choice
        )

    return openai_not_violence_function


def get_not_violencegraphic_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "violencegraphic feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_not_violencegraphic_function(prompt, response):
        return openai_moderation_not_violencegraphic(
            prompt, response, evaluation_choice
        )

    return openai_not_violencegraphic_function


def get_not_toxicity_function(provider, model_engine, evaluation_choice):
    if provider != "huggingface":
        print(
            "toxicity feedback function only has one provider. Defaulting to `provider=huggingface`"
        )

    def huggingface_not_toxic_function(prompt, response):
        return huggingface_not_toxic(prompt, response, evaluation_choice)

    return huggingface_not_toxic_function


def get_not_disinformation_function(provider, model_engine, evaluation_choice):
    if provider != "cohere":
        print(
            "disinformation feedback function only has one provider. Defaulting to `provider=cohere`"
        )

    def cohere_not_disinformation_function(prompt, response):
        return cohere_not_disinformation(prompt, response, evaluation_choice)

    return cohere_not_disinformation_function


def get_factagreement_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print(
            "violencegraphic feedback function only has one provider. Defaulting to `provider=openai`"
        )

    def openai_factagreement_function(prompt, response):
        return openai_factagreement(prompt, response, model_engine)

    return openai_factagreement_function


FEEDBACK_FUNCTIONS = {
    'sentiment-positive': get_sentimentpositive_function,
    'relevance': get_relevance_function,
    'hate': get_not_hate_function,
    'hatethreatening': get_not_hatethreatening_function,
    'selfharm': get_not_selfharm_function,
    'sexual': get_not_sexual_function,
    'sexualminors': get_not_sexualminors_function,
    'violence': get_violence_function,
    'violencegraphic': get_not_violencegraphic_function,
    'toxicity': get_not_toxicity_function,
    'disinformation': get_not_disinformation_function,
    'factagreement': get_factagreement_function,
}
