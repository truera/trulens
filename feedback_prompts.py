import cohere
from cohere.responses.classify import Example

RELEVANCE_SYSTEM_PROMPT = f"You are a relevance classifier, providing the relevance of a given response to a particular prompt. \n"
"Provide all responses only as a number from 1 to 10 where 1 is the least relevant and 10 is the most. \n"
"Never elaborate. The prompt is: "

RELEVANCE_CONTENT_PROMPT = "For that prompt, how relevant is: "

SENTIMENT_PROMPT = "Please classify the sentiment of the following text as positive or negative: "

COHERE_SENTIMENT_EXAMPLES = [
    Example("The order came 5 days early", "positive"),
    Example("The item exceeded my expectations", "positive"),
    Example("The package was damaged", "negative"),
    Example("The order is 5 days late", "negative"),
    Example("The item\'s material feels low quality", "negative"),
    Example("I used the product this morning", "neutral"),
    Example("The product arrived yesterday", "neutral"),
]
