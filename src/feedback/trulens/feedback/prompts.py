# NOTE: Try not to put anything new here. Prompts should go into
# trulens.feedback.v2.feedback unless they are related to computing
# feedback reasons which can stay here for now as there is no good place for
# those yet.

from trulens.feedback.v2 import feedback as v2

COT_REASONS_TEMPLATE = """
Please answer using the entire template below.

TEMPLATE:
Criteria: <Provide the criteria for this evaluation>
Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
Score: <The score based on the given criteria>
"""

# Keep this in line with the LLM output template as above
GROUNDEDNESS_REASON_TEMPLATE = """
Criteria: {statement_sentence}
Supporting Evidence: {supporting_evidence}
Score: {score}
"""

LLM_GROUNDEDNESS_FULL_PROMPT = """Give me the INFORMATION OVERLAP of this SOURCE and STATEMENT.
SOURCE: {premise}
STATEMENT: {hypothesis}
"""

LLM_GROUNDEDNESS_SYSTEM = v2.Groundedness.system_prompt
LLM_GROUNDEDNESS_USER = v2.Groundedness.user_prompt
LLM_GROUNDEDNESS_SENTENCES_SPLITTER = v2.Groundedness.sentences_splitter_prompt

LLM_ANSWERABILITY_SYSTEM = v2.Answerability.system_prompt
LLM_ANSWERABILITY_USER = v2.Answerability.user_prompt

LLM_ABSTENTION_SYSTEM = v2.Abstention.system_prompt
LLM_ABSTENTION_USER = v2.Abstention.user_prompt

LLM_TRIVIAL_SYSTEM = v2.Trivial.system_prompt
LLM_TRIVIAL_USER = v2.Trivial.user_prompt

CONTEXT_RELEVANCE_SYSTEM = v2.ContextRelevance.system_prompt
CONTEXT_RELEVANCE_USER = v2.ContextRelevance.user_prompt
CONTEXT_RELEVANCE_DEFAULT_COT_PROMPT = v2.ContextRelevance.default_cot_prompt

ANSWER_RELEVANCE_SYSTEM = v2.PromptResponseRelevance.system_prompt
ANSWER_RELEVANCE_USER = v2.PromptResponseRelevance.user_prompt

SYSTEM_FIND_SUPPORTING = """
You are a summarizer that can only answer 'Nothing Found' or return exact sentences from this excerpt:

{prompt}
"""

USER_FIND_SUPPORTING = """
I'm looking for related information to a statement from your excerpt. If nothing is directly related, say 'Nothing Found'
Respond with all sentences, unchanged from the excerpt, that are directly related to this statement: {response}
"""

SENTIMENT_SYSTEM = v2.Sentiment.system_prompt
SENTIMENT_USER = v2.Sentiment.user_prompt

CORRECT_SYSTEM = """
You are a fact bot and you answer with verifiable facts
"""

AGREEMENT_SYSTEM = """
You will continually start seeing responses to the prompt:

%s

The expected answer is:

%s

Answer only with an integer from 1 to 10 based on how semantically similar the responses are to the expected answer.
where 0 is no semantic similarity at all and 10 is perfect agreement between the responses and the expected answer.
On a NEW LINE, give the integer score and nothing more.
"""

REMOVE_Y_N = " If so, respond Y. If not, respond N."

LANGCHAIN_CONCISENESS_SYSTEM_PROMPT = v2.Conciseness.system_prompt

LANGCHAIN_CORRECTNESS_SYSTEM_PROMPT = v2.Correctness.system_prompt

LANGCHAIN_COHERENCE_SYSTEM_PROMPT = v2.Coherence.system_prompt

LANGCHAIN_HARMFULNESS_SYSTEM_PROMPT = v2.Harmfulness.system_prompt

LANGCHAIN_MALICIOUSNESS_SYSTEM_PROMPT = v2.Maliciousness.system_prompt

LANGCHAIN_HELPFULNESS_SYSTEM_PROMPT = v2.Helpfulness.system_prompt

LANGCHAIN_CONTROVERSIALITY_SYSTEM_PROMPT = v2.Controversiality.system_prompt

LANGCHAIN_MISOGYNY_SYSTEM_PROMPT = v2.Misogyny.system_prompt

LANGCHAIN_CRIMINALITY_SYSTEM_PROMPT = v2.Criminality.system_prompt

LANGCHAIN_INSENSITIVITY_SYSTEM_PROMPT = v2.Insensitivity.system_prompt

LANGCHAIN_PROMPT_TEMPLATE_SYSTEM = """
CRITERIA:

{criteria}
"""

LANGCHAIN_PROMPT_TEMPLATE_USER = """
SUBMISSION:

{submission}"""

LANGCHAIN_PROMPT_TEMPLATE_WITH_COT_REASONS_SYSTEM = (
    LANGCHAIN_PROMPT_TEMPLATE_SYSTEM + COT_REASONS_TEMPLATE
)

STEREOTYPES_SYSTEM_PROMPT = v2.Stereotypes.system_prompt
STEREOTYPES_USER_PROMPT = v2.Stereotypes.user_prompt

GENERATE_KEY_POINTS_SYSTEM_PROMPT = """
INSTRUCTIONS:

1. Identify the key points in the provided source text.

2. Assign each point high or low importance level.

3. Remove any points that are not assessed to high importance.

4. All key points should now be assessed to high importance. There is no need to mention a points importance level.

Answer using the entire template below. Each key point must be on a new line.

TEMPLATE:
Key Point 1: <The key point from the source text>
Key Point 2: <The key point from the source text>
Key Point 3: <The key point from the source text>
...
"""

GENERATE_KEY_POINTS_USER_PROMPT = """
/SOURCE TEXT/
{source}
/END OF SOURCE TEXT/
"""

COMPREHENSIVENESS_SYSTEM_PROMPT = v2.Comprehensiveness.system_prompt

COMPREHENSIVENESS_USER_PROMPT = v2.Comprehensiveness.user_prompt

TRAJECTORY_EVAL_STEP_RELEVANCE_SYSTEM_PROMPT = (
    v2.TrajectoryStepRelevance.system_prompt
)

TRAJECTORY_EVAL_LOGICAL_CONSISTENCY_SYSTEM_PROMPT = (
    v2.TrajectoryLogicalConsistency.system_prompt
)

TRAJECTORY_EVAL_WORKFLOW_EFFICIENCY_SYSTEM_PROMPT = (
    v2.TrajectoryWorkflowEfficiency.system_prompt
)
