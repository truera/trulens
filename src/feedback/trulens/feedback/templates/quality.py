"""
Text quality evaluation templates: coherence, conciseness,
correctness, sentiment, helpfulness, controversiality, etc.
"""

from inspect import cleandoc
from typing import ClassVar

from trulens.feedback.templates.base import LIKERT_0_3_PROMPT
from trulens.feedback.templates.base import CriteriaOutputSpaceMixin
from trulens.feedback.templates.base import OutputSpace
from trulens.feedback.templates.base import Semantics
from trulens.feedback.templates.base import WithPrompt
from trulens.feedback.templates.base import supported_criteria


class Conciseness(Semantics, WithPrompt):  # or syntax
    # openai.conciseness

    # langchain Criteria.CONCISENESS
    system_prompt: ClassVar[str] = cleandoc(
        f"""{supported_criteria["conciseness"]} Respond only as a number from {"{min_score}"} to {"{max_score}"} where {"{min_score}"} is the least concise and {"{max_score}"} is the most concise."""
    )


class Correctness(Semantics, WithPrompt):
    # openai.correctness
    # openai.correctness_with_cot_reasons

    # langchain Criteria.CORRECTNESS
    system_prompt: ClassVar[str] = cleandoc(
        f"""{supported_criteria["correctness"]} Respond only as a number from {"{min_score}"} to {"{max_score}"} where {"{min_score}"} is the least correct and {"{max_score}"} is the most correct."""
    )


class Coherence(Semantics):
    # openai.coherence
    # openai.coherence_with_cot_reasons

    system_prompt: ClassVar[str] = cleandoc(
        f"""{supported_criteria["coherence"]} Respond only as a number from {"{min_score}"} to {"{max_score}"} where {"{min_score}"} is the least coherent and {"{max_score}"} is the most coherent."""
    )


class Sentiment(Semantics, WithPrompt, CriteriaOutputSpaceMixin):
    """
    This evaluates the *positive sentiment* of either the
    prompt or response.
    """

    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = """
    - Sentiment should be evaluated based on the overall tone of the submission.
    - Negative sentiment should get a score of {min_score}.
    - Positive sentiment should get a score of {max_score}.
    """

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt_template: ClassVar[str] = cleandoc(
        """
        You are a SENTIMENT grader; providing the sentiment of the given SUBMISSION.
        Respond only as a number from {output_space_prompt}.

        Criteria for evaluating sentiment:
        {criteria}
        {additional_instructions}
        """
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )

    user_prompt: ClassVar[str] = cleandoc("""Submission: """)


class Helpfulness(Semantics, CriteriaOutputSpaceMixin):
    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = f"""
        {supported_criteria["helpfulness"]}
        """

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt_template: ClassVar[str] = cleandoc(
        """
        {criteria}
        {additional_instructions}
        """
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )

    user_prompt: ClassVar[str] = cleandoc("""Submission: """)


class Controversiality(Semantics):
    output_space_prompt: ClassVar[str] = LIKERT_0_3_PROMPT
    output_space: ClassVar[str] = OutputSpace.LIKERT_0_3.name
    criteria_template: ClassVar[str] = f"""
        {supported_criteria["controversiality"]}
        """

    criteria: ClassVar[str] = criteria_template.format(
        min_score=OutputSpace.LIKERT_0_3.value[0],
        max_score=OutputSpace.LIKERT_0_3.value[1],
    )

    system_prompt_template: ClassVar[str] = cleandoc(
        """
        {criteria}
        {additional_instructions}
        """
    )

    system_prompt: ClassVar[str] = cleandoc(
        system_prompt_template.format(
            output_space_prompt=output_space_prompt,
            criteria=criteria,
            additional_instructions="",
        )
    )

    user_prompt: ClassVar[str] = cleandoc("""Submission: """)


# ------------------------------------------------------------------
# Standalone prompt constants for quality evaluation
# ------------------------------------------------------------------

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
