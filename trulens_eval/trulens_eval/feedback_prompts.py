from cohere.responses.classify import Example

LLM_GROUNDEDNESS = """You are a INFORMATION OVERLAP classifier; providing the overlap of information between two statements.
Respond only as a number from 1 to 10 where 1 is no information overlap and 10 is all information is overlapping.
Never elaborate.

STATEMENT 1: {premise}

STATEMENT 2: {hypothesis}

INFORMATION OVERLAP: """

LLM_GROUNDEDNESS_FULL_SYSTEM = """You are a INFORMATION OVERLAP classifier providing the overlap of information between a SOURCE and STATEMENT.
For every sentence in the statement, please answer with this template:

TEMPLATE: 
Statement Sentence: <Sentence>, 
Supporting Evidence: <Choose the exact unchanged sentences in the source that can answer the statement, if nothing matches, say NOTHING FOUND>
Score: <Output a number between 1-10 where 1 is no information overlap and 10 is all information is overlapping.
"""

# Keep this in line with the LLM output template as above
GROUNDEDNESS_REASON_TEMPLATE = """
Statement Sentence: {statement_sentence} 
Supporting Evidence: {supporting_evidence} 
Score: {score} 

"""

LLM_GROUNDEDNESS_FULL_PROMPT = """Give me the INFORMATION OVERLAP of this SOURCE and STATEMENT.

SOURCE: {premise}

STATEMENT: {hypothesis}
"""

QS_RELEVANCE = """You are a RELEVANCE grader; providing the relevance of the given STATEMENT to the given QUESTION.
Respond only as a number from 1 to 10 where 1 is the least relevant and 10 is the most relevant. 

A few additional scoring guidelines:

- Long STATEMENTS should score equally well as short STATEMENTS.

- RELEVANCE score should increase as the STATEMENT provides more RELEVANT context to the QUESTION.

- RELEVANCE score should increase as the STATEMENT provides RELEVANT context to more parts of the QUESTION.

- STATEMENT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

- STATEMENT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

- STATEMENT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

- STATEMENT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

- Answers that intentionally do not answer the question, such as 'I don't know', should also be counted as the most relevant.

- Never elaborate.

QUESTION: {question}

STATEMENT: {statement}

RELEVANCE: """

PR_RELEVANCE = """You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given PROMPT.
Respond only as a number from 1 to 10 where 1 is the least relevant and 10 is the most relevant. 

A few additional scoring guidelines:

- Long RESPONSES should score equally well as short RESPONSES.

- Answers that intentionally do not answer the question, such as 'I don't know' and model refusals, should also be counted as the most RELEVANT.

- RESPONSE must be relevant to the entire PROMPT to get a score of 10.

- RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the PROMPT.

- RESPONSE that is RELEVANT to none of the PROMPT should get a score of 1.

- RESPONSE that is RELEVANT to some of the PROMPT should get as score of 2, 3, or 4. Higher score indicates more RELEVANCE.

- RESPONSE that is RELEVANT to most of the PROMPT should get a score between a 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

- RESPONSE that is RELEVANT to the entire PROMPT should get a score of 9 or 10.

- RESPONSE that is RELEVANT and answers the entire PROMPT completely should get a score of 10.

- RESPONSE that confidently FALSE should get a score of 1.

- RESPONSE that is only seemingly RELEVANT should get a score of 1.

- Never elaborate.

PROMPT: {prompt}

RESPONSE: {response}

RELEVANCE: """

SYSTEM_FIND_SUPPORTING = """
You are a summarizer that can only answer 'Nothing Found' or return exact sentences from this excerpt:

{prompt}
"""

USER_FIND_SUPPORTING = """
I'm looking for related information to a statement from your excerpt. If nothing is directly related, say 'Nothing Found'
Respond with all sentences, unchanged from the excerpt, that are directly related to this statement: {response}
"""

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
