import openai
import re

def relevance_function(prompt, response):
    return re.search('[0-9]+', openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 0.5,
    messages=[
            {"role": "system", "content": "You are a relevance classifier, providing the relevance to this text: " + prompt + " Provide all responses only as a number from 0 to 9. Never elaborate."},
            {"role": "user", "content": "Rate the relevance of the following piece of text:" + response}
        ]
    )["choices"][0]["message"]["content"]).group()


def sentiment_function(prompt, response):
    return re.search('[0-9]+', openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 0.5,
    messages=[
            {"role": "system", "content": "You are a sentiment classifier, providing the relevance to this text: " + prompt + " Provide all responses only as a number from 0 to 9. Never elaborate."},
            {"role": "user", "content": "Rate the sentiment of the following piece of text:" + response}
        ]
    )["choices"][0]["message"]["content"]).group()


FEEDBACK_FUNCTIONS = {
    'relevance': relevance_function,
    'sentiment': sentiment_function, 
}