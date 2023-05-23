from cohere.responses.classify import Example

QS_RELEVANCE = """You are a RELEVANCE classifier; providing the relevance of the given STATEMENT to the given QUESTION.
Respond only as a number from 1 to 10 where 1 is the least relevant and 10 is the most relevant.
Never elaborate.

QUESTION: {question}

STATEMENT: {statement}

RELEVANCE: """

PR_RELEVANCE = """
You are a relevance classifier, providing the relevance of a given response to the given prompt.
Respond only as a number from 1 to 10 where 1 is the least relevant and 10 is the most relevant.
Never elaborate.

Prompt: {prompt}

Response: {response}

Relevance: """

SENTIMENT_SYSTEM_PROMPT = f"Please classify the sentiment of the following text as 1 if positive or 0 if not positive. Respond with only a '1' or '0', nothing more."
RELEVANCE_SYSTEM_PROMPT = f"You are a relevance classifier, providing the relevance of a given response to a particular prompt. \n"
"Provide all responses only as a number from 1 to 10 where 1 is the least relevant and 10 is the most. Always respond with an integer between 1 and 10. \n"
"Never elaborate. The prompt is: "
RELEVANCE_CONTENT_PROMPT = f"For that prompt, how relevant is this response on the scale between 1 and 10: "

COHERE_SENTIMENT_EXAMPLES = [
    Example("The order came 5 days early", "1"),
    Example("I just got a promotion at work and I\'m so excited!", "1"),
    Example(
        "My best friend surprised me with tickets to my favorite band's concert.",
        "1"
    ),
    Example(
        "I\'m so grateful for my family's support during a difficult time.", "1"
    ),
    Example("It\'s kind of grungy, but the pumpkin pie slaps", "1"),
    Example(
        "I love spending time in nature and feeling connected to the earth.",
        "1"
    ),
    Example("I had an amazing meal at the new restaurant in town", "1"),
    Example("The pizza is good, but the staff is horrible to us", "0"),
    Example("The package was damaged", "0"),
    Example("I\'m feeling really sick and can\'t seem to shake it off", "0"),
    Example("I got into a car accident and my car is completely totaled.", "0"),
    Example(
        "My boss gave me a bad performance review and I might get fired", "0"
    ),
    Example("I got into a car accident and my car is completely totaled.", "0"),
    Example(
        "I\'m so disapointed in myself for not following through on my goals",
        "0"
    )
]

COHERE_NOT_DISINFORMATION_EXAMPLES = [
    Example(
        "Bud Light Official SALES REPORT Just Released ′ 50% DROP In Sales ′ Total COLLAPSE ′ Bankruptcy?",
        "0"
    ),
    Example(
        "The Centers for Disease Control and Prevention quietly confirmed that at least 118,000 children and young adults have “died suddenly” in the U.S. since the COVID-19 vaccines rolled out,",
        "0"
    ),
    Example(
        "Silicon Valley Bank collapses, in biggest failure since financial crisis",
        "1"
    ),
    Example(
        "Biden admin says Alabama health officials didn’t address sewage system failures disproportionately affecting Black residents",
        "1"
    )
]


CORRECT_SYSTEM_PROMPT = \
""" 
You are a fact bot and you answer with verifiable facts
"""

AGREEMENT_SYSTEM_PROMPT = \
""" 
You will continually start seeing responses to the prompt:

%s

The right answer is:

%s

Answer only with an integer from 1 to 10 based on how close the responses are to the right answer.
"""
