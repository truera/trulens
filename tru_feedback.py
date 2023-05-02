import re

import openai


def relevance_function(prompt, response):
    return re.search(
        '[0-9]+',
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            messages=[{
                "role":
                "system",
                "content":
                "You are a relevance classifier, providing the relevance to this text: "
                + prompt +
                " Provide all responses only as a number from 0 to 9. Never elaborate."
            }, {
                "role":
                "user",
                "content":
                "Rate the relevance of the following piece of text:" + response
            }])["choices"][0]["message"]["content"]).group()


def sentiment_function(prompt, response):
    model_engine = "text-davinci-002"
    prompt = (
        f"Please classify the sentiment of the following text: \"{response}\" as one of the following:\n"
        "Positive\n"
        "Negative\n"
        "Classify the sentiment:")

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.5,
    )

    sentiment = response.choices[0].text.strip().lower()

    if sentiment == "positive":
        return 1
    else:
        return 0


FEEDBACK_FUNCTIONS = {
    'relevance': relevance_function,
    'sentiment': sentiment_function,
}
