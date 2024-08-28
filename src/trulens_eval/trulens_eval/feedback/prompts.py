# ruff: noqa: E402, F403, F401
"""
!!! warning
    This module is deprecated and will be removed. Use
    `trulens.feedback.prompts` instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.feedback.prompts import AGREEMENT_SYSTEM
from trulens.feedback.prompts import ANSWER_RELEVANCE_SYSTEM
from trulens.feedback.prompts import ANSWER_RELEVANCE_USER
from trulens.feedback.prompts import COMPREHENSIVENESS_SYSTEM_PROMPT
from trulens.feedback.prompts import COMPREHENSIVENESS_USER_PROMPT
from trulens.feedback.prompts import CONTEXT_RELEVANCE_SYSTEM
from trulens.feedback.prompts import CONTEXT_RELEVANCE_USER
from trulens.feedback.prompts import CORRECT_SYSTEM
from trulens.feedback.prompts import COT_REASONS_TEMPLATE
from trulens.feedback.prompts import GENERATE_KEY_POINTS_SYSTEM_PROMPT
from trulens.feedback.prompts import GENERATE_KEY_POINTS_USER_PROMPT
from trulens.feedback.prompts import GROUNDEDNESS_REASON_TEMPLATE
from trulens.feedback.prompts import LANGCHAIN_COHERENCE_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_CONCISENESS_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_CONTROVERSIALITY_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_CORRECTNESS_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_CRIMINALITY_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_HARMFULNESS_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_HELPFULNESS_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_INSENSITIVITY_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_MALICIOUSNESS_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_MISOGYNY_SYSTEM_PROMPT
from trulens.feedback.prompts import LANGCHAIN_PROMPT_TEMPLATE_SYSTEM
from trulens.feedback.prompts import LANGCHAIN_PROMPT_TEMPLATE_USER
from trulens.feedback.prompts import (
    LANGCHAIN_PROMPT_TEMPLATE_WITH_COT_REASONS_SYSTEM,
)
from trulens.feedback.prompts import LLM_GROUNDEDNESS_FULL_PROMPT
from trulens.feedback.prompts import LLM_GROUNDEDNESS_SYSTEM
from trulens.feedback.prompts import LLM_GROUNDEDNESS_USER
from trulens.feedback.prompts import REMOVE_Y_N
from trulens.feedback.prompts import SENTIMENT_SYSTEM
from trulens.feedback.prompts import SENTIMENT_USER
from trulens.feedback.prompts import STEREOTYPES_SYSTEM_PROMPT
from trulens.feedback.prompts import STEREOTYPES_USER_PROMPT
from trulens.feedback.prompts import SYSTEM_FIND_SUPPORTING
from trulens.feedback.prompts import USER_FIND_SUPPORTING

QS_RELEVANCE_VERB_2S_TOP1: str = deprecation_utils.deprecated_str(
    "QS_RELEVANCE_VERB_2S_TOP1",
    reason="QS_RELEVANCE was removed in favor of ANSWER_RELEVANCE or CONTEXT_RELEVANCE",
)
