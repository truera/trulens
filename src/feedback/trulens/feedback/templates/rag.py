"""
RAG evaluation templates: groundedness, context relevance,
answer relevance, answerability, comprehensiveness, etc.
"""

from inspect import cleandoc
from typing import ClassVar

from trulens.feedback.templates.base import LIKERT_0_3_PROMPT
from trulens.feedback.templates.base import CriteriaOutputSpaceMixin
from trulens.feedback.templates.base import OutputSpace
from trulens.feedback.templates.base import Semantics
from trulens.feedback.templates.base import WithPrompt


class GroundTruth(Semantics):
    # Some groundtruth may also be syntactic if it merely
    # compares strings without interpretation by some model
    # like these below:

    # GroundTruthAgreement.bert_score
    # GroundTruthAgreement.bleu
    # GroundTruthAgreement.rouge
    # GroundTruthAgreement.agreement_measure
    pass


class Relevance(Semantics):
    """
    This evaluates the *relevance* of the LLM response to
    the given text by LLM prompting.

    Relevance is available for any LLM provider.
    """

    # openai.relevance
    # openai.relevance_with_cot_reasons
    pass


class Groundedness(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """
    - Statements that are directly supported by the source should be considered grounded and should get a high score.
    - Statements that are not directly supported by the source should be considered not grounded and should get a low score.
    - Statements of doubt, that admissions of uncertainty or not knowing the answer are considered abstention, and should be counted as the most overlap and therefore get a max score of {max_score}.
    - Consider indirect or implicit evidence, or the context of the statement, to avoid penalizing potentially factual claims due to lack of explicit support.
    - Be cautious of false positives; ensure that high scores are only given when there is clear supporting evidence.
    - Pay special attention to ensure that indirect evidence is not mistaken for direct support.
    """

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are an INFORMATION OVERLAP classifier; providing the overlap of information (entailment or groundedness) between the source and statement.

        Respond only as a number from {output_space_prompt}.

        You should score the groundedness of the statement based on the following criteria:
        {criteria}
        {additional_instructions}

        Never elaborate."""
    )

    user_prompt: ClassVar[str] = cleandoc(
        """SOURCE: {premise}

        STATEMENT: {hypothesis}

        Respond ONLY with a single-line JSON object having exactly these keys:
          "criteria"             – copy of the STATEMENT verbatim
          "supporting_evidence"  – sentence(s) from SOURCE that support the STATEMENT, or the string NOTHING FOUND or ABSTENTION
          "score"                – an integer 0, 1, 2, or 3

        Example (format only – you must replace the values):
        {{"criteria": "...", "supporting_evidence": "...", "score": 2}}

        Return the JSON and nothing else (no markdown, no additional text)."""
    )

    sentences_splitter_prompt: ClassVar[str] = cleandoc(
        """Split the following statement into individual sentences:

        Statement: {statement}

        Return each sentence on a new line.
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )


class Answerability(Semantics, WithPrompt):
    system_prompt: ClassVar[str] = cleandoc(
        """You are a ANSWERABILITY classifier; providing a score of {min_score} if the answer to the QUESTION does not exist in the SOURCE, and a {max_score} if the answer does exist in the SOURCE.
        Do not consider the quality of the answer, only if it exists or not.
        Never elaborate."""
    )
    user_prompt: ClassVar[str] = cleandoc(
        """QUESTION: {question}

        SOURCE: {source}

        ANSWERABILITY:"""
    )


class Abstention(Semantics, WithPrompt):
    system_prompt: ClassVar[str] = cleandoc(
        """You are a ABSTENTION classifier; classifying the STATEMENT as an abstention or not.
        Examples of an abstention include statement similar to 'I don't know' or 'I can't answer that'.
        Respond only as a number from {min_score} to {max_score} where {min_score} is not an abstention and {max_score} is an abstention.
        Never elaborate."""
    )
    user_prompt: ClassVar[str] = cleandoc(
        """STATEMENT: {statement}

        ABSTENTION:"""
    )


class Trivial(Semantics, WithPrompt):
    system_prompt: ClassVar[str] = cleandoc(
        """Consider the following list of statements. Identify and remove sentences that are stylistic, contain trivial pleasantries, or lack substantive information relevant to the main content. Respond only with a list of the remaining statements in the format of a python list of strings."""
    )
    user_prompt: ClassVar[str] = cleandoc(
        """ALL STATEMENTS: {statements}

        IMPORTANT STATEMENTS: """
    )


class ContextRelevance(Relevance, WithPrompt, CriteriaOutputSpaceMixin):
    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """

        - SEARCH RESULT that is IRRELEVANT to the USER QUERY should score {min_score}.
        - SEARCH RESULT that is RELEVANT to some of the USER QUERY should get an intermediate score.
        - SEARCH RESULT that is RELEVANT to most of the USER QUERY should get a score closer to {max_score}.
        - SEARCH RESULT that is RELEVANT to the entirety of the USER QUERY should get a score of {max_score}, which is the full mark.
        - SEARCH RESULT must be relevant and helpful for answering the entire USER QUERY to get a score of {max_score}.
        """

    default_cot_prompt: ClassVar[str] = cleandoc(
        """You are an EXPERT SEARCH RESULT RATER. You are given a USER QUERY and a SEARCH RESULT.
        Your task is to rate the search result based on its relevance to the user query. You should rate the search result on a scale of 0 to 3, where:
        0: The search result has no relevance to the user query.
        1: The search result has low relevance to the user query. It may contain some information that is very slightly related to the user query but not enough to answer it. The search result contains some references or very limited information about some entities present in the user query. In case the query is a statement on a topic, the search result should be tangentially related to it.
        2: The search result has medium relevance to the user query. If the user query is a question, the search result may contain some information that is relevant to the user query but not enough to answer it. If the user query is a search phrase/sentence, either the search result is centered around most but not all entities present in the user query, or if all the entities are present in the result, the search result while not being centered around it has medium level of relevance. In case the query is a statement on a topic, the search result should be related to the topic.
        3: The search result has high relevance to the user query. If the user query is a question, the search result contains information that can answer the user query. Otherwise, if the search query is a search phrase/sentence, it provides relevant information about all entities that are present in the user query and the search result is centered around the entities mentioned in the query. In case the query is a statement on a topic, the search result should be either directly addressing it or be on the same topic.

        You should think step by step about the user query and the search result and rate the search result. Be critical and strict with your ratings to ensure accuracy.

        Think step by step about the user query and the search result and rate the search result. Provide a reasoning for your rating.

        """
    )

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are a RELEVANCE grader; providing the relevance of the given USER QUERY to the given SEARCH RESULT.
        Respond only as a number from {output_space_prompt}.

        Criteria for evaluating relevance:
        {criteria}
        {additional_instructions}

        A few additional scoring guidelines:

        - Long SEARCH RESULT should score equally well as short SEARCH RESULT.

        - RELEVANCE score should increase as the SEARCH RESULT provides more RELEVANT context to the USER QUERY.

        - RELEVANCE score should increase as the SEARCH RESULT provides RELEVANT context to more parts of the USER QUERY.

        - Never elaborate.
        """
    )

    user_prompt: ClassVar[str] = cleandoc(
        """USER QUERY: {question}
        SEARCH RESULT: {context}

        RELEVANCE:
        """
    )

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )


class PromptResponseRelevance(Relevance, WithPrompt, CriteriaOutputSpaceMixin):
    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """
        - RESPONSE must be relevant to the entire PROMPT to get a maximum score of {max_score}.
        - RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the PROMPT.
        - RESPONSE that is RELEVANT to none of the PROMPT should get a minimum score of {min_score}.
        - RESPONSE that is RELEVANT and answers the entire PROMPT completely should get a score of {max_score}.
        - RESPONSE that confidently FALSE should get a score of {min_score}.
        - RESPONSE that is only seemingly RELEVANT should get a score of {min_score}.
        - Long RESPONSES should score equally well as short RESPONSES.
        - Answers that intentionally do not answer the question, such as 'I don't know' and model refusals, should also be counted as the least RELEVANT and get a score of {min_score}.
    """

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt_template: ClassVar[str] = cleandoc(
        """
        You are a RELEVANCE grader; evaluate how relevant the RESPONSE is to the PROMPT.

        Criteria for evaluating relevance:
        {criteria}
        {additional_instructions}

        Output only a single-line JSON object with exactly these keys:
          "criteria"             – One concise sentence that states your rationale with reference to the rubric. If your criteria includes additional instructions, repeat them here.
          "supporting_evidence"  – An explanation of why you scored the way you did using exact words or evidence from the response
          "score"                – {output_space_prompt}
        """
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )

    user_prompt: ClassVar[str] = cleandoc(
        """
        PROMPT: {prompt}

        RESPONSE: {response}

        RELEVANCE:

        Produce the JSON object now.
        """
    )


class Comprehensiveness(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """
    - {min_score} - The key point is not included in the summary.
    - A middle score - The key point is vaguely mentioned or partially included in the summary.
    - {max_score} - The key point is fully included in the summary.
    """

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt_template: ClassVar[str] = cleandoc(
        """You are tasked with evaluating summarization quality. Please follow the instructions below.

        INSTRUCTIONS:

        1. Given a key point, score well the summary captures that key points.

        Are the key points from the source text comprehensively included in the summary? More important key points matter more in the evaluation.

        Scoring criteria:
        {criteria}
        {additional_instructions}

        Answer using the entire template below.

        TEMPLATE:
        Score: {output_space_prompt}
        Key Point: <Mention the key point from the source text being evaluated>
        Supporting Evidence: <Evidence of whether the key point is present or absent in the summary.>
        """
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )

    user_prompt: ClassVar[str] = cleandoc(
        """
        /KEY POINT/
        {key_point}
        /END OF KEY POINT/

        /SUMMARY/
        {summary}
        /END OF SUMMARY/
        """
    )


# ------------------------------------------------------------------
# Standalone prompt constants for RAG evaluation
# ------------------------------------------------------------------

SYSTEM_FIND_SUPPORTING = """
You are a summarizer that can only answer 'Nothing Found' or return exact sentences from this excerpt:

{prompt}
"""

USER_FIND_SUPPORTING = """
I'm looking for related information to a statement from your excerpt. If nothing is directly related, say 'Nothing Found'
Respond with all sentences, unchanged from the excerpt, that are directly related to this statement: {response}
"""

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

# Python str.format() template for formatting NLI-based
# groundedness results (used by HuggingFace provider).
# This is NOT an LLM prompt.
GROUNDEDNESS_NLI_REASON_FORMAT = """
Criteria: {statement_sentence}
Supporting Evidence: {supporting_evidence}
Score: {score}
"""

# Backward-compat alias (old name).
GROUNDEDNESS_REASON_TEMPLATE = GROUNDEDNESS_NLI_REASON_FORMAT
