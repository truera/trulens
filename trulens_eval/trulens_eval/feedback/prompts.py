# NOTE: Try not to put anything new here. Prompts should go into
# trulens_eval.feedback.v2.feedback unless they are related to computing
# feedback reasons which can stay here for now as there is no good place for
# those yet.

from trulens_eval.feedback.v2 import feedback as v2

COT_REASONS_TEMPLATE = \
"""
Please answer with this template:

TEMPLATE: 
Criteria: <Provide the criteria for this evaluation>
Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
Score: <The score 0-10 based on the given criteria>
"""

LLM_GROUNDEDNESS = v2.Groundedness.prompt.template

LLM_GROUNDEDNESS_SYSTEM_NO_COT = """You are a INFORMATION OVERLAP classifier providing the overlap of information between a SOURCE and STATEMENT.

Output a number between 0-10 where 0 is no information overlap and 10 is all information is overlapping. Never elaborate.
"""

LLM_GROUNDEDNESS_FULL_SYSTEM = """You are a INFORMATION OVERLAP classifier providing the overlap of information between a SOURCE and STATEMENT.
For every sentence in the statement, please answer with this template:

TEMPLATE: 
Statement Sentence: <Sentence>, 
Supporting Evidence: <Choose the exact unchanged sentences in the source that can answer the statement, if nothing matches, say NOTHING FOUND>
Score: <Output a number between 0-10 where 0 is no information overlap and 10 is all information is overlapping>
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

QS_RELEVANCE = v2.QuestionStatementRelevance.prompt.template
PR_RELEVANCE = v2.PromptResponseRelevance.prompt.template

SYSTEM_FIND_SUPPORTING = """
You are a summarizer that can only answer 'Nothing Found' or return exact sentences from this excerpt:

{prompt}
"""

USER_FIND_SUPPORTING = """
I'm looking for related information to a statement from your excerpt. If nothing is directly related, say 'Nothing Found'
Respond with all sentences, unchanged from the excerpt, that are directly related to this statement: {response}
"""

SENTIMENT_SYSTEM_PROMPT = v2.Sentiment.prompt.template

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

REMOVE_Y_N = " If so, respond Y. If not, respond N."

LANGCHAIN_CONCISENESS_PROMPT = v2.Conciseness.prompt.template
LANGCHAIN_CORRECTNESS_PROMPT = v2.Correctness.prompt.template
LANGCHAIN_COHERENCE_PROMPT = v2.Coherence.prompt.template
LANGCHAIN_HARMFULNESS_PROMPT = v2.Harmfulness.prompt.template.replace(
    REMOVE_Y_N, ""
)
LANGCHAIN_MALICIOUSNESS_PROMPT = v2.Maliciousness.prompt.template.replace(
    REMOVE_Y_N, ""
)
LANGCHAIN_HELPFULNESS_PROMPT = v2.Helpfulness.prompt.template.replace(
    REMOVE_Y_N, ""
)
LANGCHAIN_CONTROVERSIALITY_PROMPT = v2.Controversiality.prompt.template.replace(
    REMOVE_Y_N, ""
)
LANGCHAIN_MISOGYNY_PROMPT = v2.Misogyny.prompt.template.replace(REMOVE_Y_N, "")
LANGCHAIN_CRIMINALITY_PROMPT = v2.Criminality.prompt.template.replace(
    REMOVE_Y_N, ""
)
LANGCHAIN_INSENSITIVITY_PROMPT = v2.Insensitivity.prompt.template.replace(
    REMOVE_Y_N, ""
)

LANGCHAIN_PROMPT_TEMPLATE = """
CRITERIA:

{criteria}

SUBMISSION:

{submission}
"""

LANGCHAIN_PROMPT_TEMPLATE_WITH_COT_REASONS = LANGCHAIN_PROMPT_TEMPLATE + COT_REASONS_TEMPLATE

STEREOTYPES_PROMPT = v2.Stereotypes.prompt.template

SUMMARIZATION_PROMPT = """
You are a helper for summarization tasks. You will do two steps. You must do both steps, and most importantly, you must add a score.

STEP 1 - Most Important Points:
(DO NOT SKIP THIS STEP!)

For the given SOURCE What are the most important points?

please answer with this template:

(Step 1)
Important Points: <Outline the important points>

Step 2 - Compare:
(DO NOT SKIP THIS STEP!)

For a SUMMARY How well does this summary address the above main points?

please answer with this template:

(Step 2)
Supporting Evidence: <For each of the Important Points, explain if the SUMMARY does or does not mention it.>
Score: <Give a score from 0 to 10 on if the SUMMARY addresses every single one of the main points. A score of 0 is no points were mentioned. A score of 5 is half the points were mentioned. a score of 10 is all points were mentioned.>


/START SUMMARY/ 
{summary}
/END SUMMARY/ 

/START SOURCE/ 
{source}
/END SOURCE/ 
"""
