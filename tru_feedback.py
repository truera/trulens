import re

import cohere
from cohere.responses.classify import Example
import dotenv
import openai
import requests

import feedback_prompts
from keys import *

# openai


def openai_moderation_hate(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = openai.Moderation.create(input=input)
    return int(openai_response["results"][0]["categories"]["hate"])


def openai_moderation_hatethreatening(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = openai.Moderation.create(input=input)
    return int(openai_response["results"][0]["categories"]["hate/threatening"])


def openai_moderation_selfharm(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = openai.Moderation.create(input=input)
    return int(openai_response["results"][0]["categories"]["self-harm"])


def openai_moderation_sexual(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = openai.Moderation.create(input=input)
    return int(openai_response["results"][0]["categories"]["sexual"])


def openai_moderation_sexualminors(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = openai.Moderation.create(input=input)
    return int(openai_response["results"][0]["categories"]["sexual/minors"])


def openai_moderation_violence(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = openai.Moderation.create(input=input)
    return int(openai_response["results"][0]["categories"]["violence"])


def openai_moderation_violencegraphic(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    openai_response = openai.Moderation.create(input=input)
    return int(openai_response["results"][0]["categories"]["violence/graphic"])


def openai_relevance(prompt, response, model_engine):

    return int(
        re.search(
            '[1-10]+',
            openai.ChatCompletion.create(
                model=model_engine,
                temperature=0.5,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            feedback_prompts.RELEVANCE_SYSTEM_PROMPT + prompt
                    }, {
                        "role":
                            "user",
                        "content":
                            feedback_prompts.RELEVANCE_CONTENT_PROMPT + response
                    }
                ]
            )["choices"][0]["message"]["content"]
        ).group()
    )


def openai_sentiment_function(
    prompt, response, evaluation_choice, model_engine
):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response

    return int(
        re.search(
            '[1-10]+',
            openai.ChatCompletion.create(
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
        ).group()
    )


# huggingface

SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
TOXIC_API_URL = "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"


def huggingface_positive_sentiment(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    max_length = 500
    truncated_text = input[:max_length]
    payload = {"inputs": truncated_text}
    hf_response = requests.post(
        SENTIMENT_API_URL, headers=HUGGINGFACE_HEADERS, json=payload
    ).json()[0]
    for label in hf_response:
        if label['label'] == 'LABEL_2':
            if label['score'] >= 0.5:
                return 1
            else:
                return 0


def huggingface_negative_sentiment(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    max_length = 500
    truncated_text = input[:max_length]
    payload = {"inputs": truncated_text}
    hf_response = requests.post(
        SENTIMENT_API_URL, headers=HUGGINGFACE_HEADERS, json=payload
    ).json()[0]
    for label in hf_response:
        if label['label'] == 'LABEL_0':
            if label['score'] >= 0.5:
                return 1
            else:
                return 0


def huggingface_toxicity(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    max_length = 500
    truncated_text = input[:max_length]
    payload = {"inputs": truncated_text}
    hf_response = requests.post(
        TOXIC_API_URL, headers=HUGGINGFACE_HEADERS, json=payload
    ).json()[0]
    for label in hf_response:
        if label['label'] == 'toxic':
            if label['score'] >= 0.5:
                return 1
            else:
                return 0


# cohere

cohere.api_key = config['COHERE_API_KEY']

co = cohere.Client(cohere.api_key)


def cohere_sentiment(prompt, response, evaluation_choice, model_engine):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    return int(
        co.classify(
            model=model_engine,
            inputs=[input],
            examples=feedback_prompts.COHERE_SENTIMENT_EXAMPLES
        )[0].prediction
    )


def cohere_disinformation(prompt, response, evaluation_choice):
    if evaluation_choice == "prompt":
        input = prompt
    if evaluation_choice == "response":
        input = response
    return int(
        co.classify(
            model='large',
            inputs=[input],
            examples=feedback_prompts.COHERE_DISINFORMATION_EXAMPLES
        )[0].prediction
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
        raise NotImplementedError("Invalid provider specified. sentiment feedback function is only supported for `provider` as `openai`, `huggingface`, or `cohere`")


def get_relevance_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print("relevance feedback function only has one provider. Defaulting to `provider=openai`")


    def openai_relevance_function(prompt, response):
        return openai_relevance(
            prompt, response, model_engine
        )

    return openai_relevance_function


def get_hate_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print("hate feedback function only has one provider. Defaulting to `provider=openai`")

    def openai_hate_function(prompt, response):
        return openai_moderation_hate(prompt, response, evaluation_choice)

    return openai_hate_function
   

def get_hatethreatening_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print("hatethreatening feedback function only has one provider. Defaulting to `provider=openai`")

    def openai_hatethreatening_function(prompt, response):
        return openai_moderation_hatethreatening(
            prompt, response, evaluation_choice
        )

    return openai_hatethreatening_function
    

def get_selfharm_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print("selfharm feedback function only has one provider. Defaulting to `provider=openai`")

    def openai_selfharm_function(prompt, response):
        return openai_moderation_selfharm(
            prompt, response, evaluation_choice
        )

    return openai_selfharm_function
    

def get_sexual_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print("sexual feedback function only has one provider. Defaulting to `provider=openai`")

    def openai_sexual_function(prompt, response):
        return openai_moderation_sexual(prompt, response, evaluation_choice)

    return openai_sexual_function


def get_sexualminors_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print("sexualminors feedback function only has one provider. Defaulting to `provider=openai`")
    
    def openai_sexualminors_function(prompt, response):
        return openai_moderation_sexual(prompt, response, evaluation_choice)

    return openai_sexualminors_function
   

def get_violence_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print("violence feedback function only has one provider. Defaulting to `provider=openai`")

    def openai_violence_function(prompt, response):
        return openai_moderation_violence(
            prompt, response, evaluation_choice
        )

    return openai_violence_function
   

def get_violencegraphic_function(provider, model_engine, evaluation_choice):
    if provider != "openai":
        print("violencegraphic feedback function only has one provider. Defaulting to `provider=openai`")

    def openai_violencegraphic_function(prompt, response):
        return openai_moderation_violencegraphic(
            prompt, response, evaluation_choice
        )

    return openai_violencegraphic_function
    

def get_toxicity_function(provider, model_engine, evaluation_choice):
    if provider != "huggingface":
        print("toxicity feedback function only has one provider. Defaulting to `provider=huggingface`")

    def huggingface_toxicity_function(prompt, response):
        return huggingface_toxicity(prompt, response, evaluation_choice)

    return huggingface_toxicity_function
    

def get_disinformation_function(provider, model_engine, evaluation_choice):
    if provider != "cohere":
        print("disinformation feedback function only has one provider. Defaulting to `provider=cohere`")

    return lambda prompt, response: cohere_disinformation(
        prompt, response, evaluation_choice
    )


FEEDBACK_FUNCTIONS = {
    'sentiment-positive': get_sentimentpositive_function,
    'relevance': get_relevance_function,
    'hate': get_hate_function,
    'hatethreatening': get_hatethreatening_function,
    'selfharm': get_selfharm_function,
    'sexual': get_sexual_function,
    'sexualminors': get_sexualminors_function,
    'violence': get_violence_function,
    'violencegraphic': get_violencegraphic_function,
    'toxicity': get_toxicity_function,
    'disinformation': get_disinformation_function,
}
