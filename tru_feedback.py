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


def openai_relevance_function(prompt, response, model_engine):
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


def sentimentpositive(
    prompt, response, evaluation_choice, provider, model_engine
):
    if provider == "openai":
        return openai_sentiment_function(
            prompt, response, evaluation_choice, model_engine
        )
    elif provider == "huggingface":
        return huggingface_positive_sentiment(
            prompt, response, evaluation_choice
        )
    elif provider == "cohere":
        return cohere_sentiment(
            prompt, response, evaluation_choice, model_engine="large"
        )


def relevance(prompt, response, evaluation_choice, provider, model_engine):
    if provider == "openai":
        return openai_relevance_function(
            prompt, response, evaluation_choice, model_engine
        )


def hate(prompt, response, evaluation_choice, provider, model_engine):
    if provider == "openai" and model_engine == "moderation":
        return openai_moderation_hate(prompt, response, evaluation_choice)


def hatethreatening(
    prompt, response, evaluation_choice, provider, model_engine
):
    if provider == "openai" and model_engine == "moderation":
        return openai_moderation_hatethreatening(
            prompt, response, evaluation_choice
        )


def selfharm(
    prompt,
    response,
    evaluation_choice,
    provider,
    model_engine,
):
    if provider == "openai" and model_engine == "moderation":
        return openai_moderation_selfharm(prompt, response, evaluation_choice)


def sexual(
    prompt,
    response,
    evaluation_choice,
    provider,
    model_engine,
):
    if provider == "openai" and model_engine == "moderation":
        return openai_moderation_sexual(prompt, response, evaluation_choice)


def sexualminors(
    prompt,
    response,
    evaluation_choice,
    provider,
    model_engine,
):
    if provider == "openai" and model_engine == "moderation":
        return openai_moderation_sexualminors(
            prompt, response, evaluation_choice
        )


def violence(
    prompt,
    response,
    evaluation_choice,
    provider,
    model_engine,
):
    if provider == "openai" and model_engine == "moderation":
        return openai_moderation_violence(prompt, response, evaluation_choice)


def violencegraphic(
    prompt,
    response,
    evaluation_choice,
    provider,
    model_engine,
):
    if provider == "openai" and model_engine == "moderation":
        return openai_moderation_violencegraphic(
            prompt, response, evaluation_choice
        )


def toxicity(
    prompt,
    response,
    evaluation_choice,
    provider,
    model_engine,
):
    if provider == "huggingface":
        return huggingface_toxicity(prompt, response, evaluation_choice)


def disinformation(
    prompt,
    response,
    evaluation_choice,
    provider,
    model_engine,
):
    if provider == "cohere":
        return cohere_disinformation(prompt, response, evaluation_choice)


FEEDBACK_FUNCTIONS = {
    'sentiment-positive': sentimentpositive,
    'relevance': relevance,
    'hate': hate,
    'hatethreatening': hatethreatening,
    'selfharm': selfharm,
    'sexual': sexual,
    'sexualminors': sexualminors,
    'violence': violence,
    'violencegraphic': violencegraphic,
    'toxicicity': toxicity,
    'disinformation': disinformation,
}
