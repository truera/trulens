# ruff: noqa: F401
"""
BACKWARD-COMPATIBILITY SHIM.

All prompt constants and template classes have moved to
``trulens.feedback.templates`` (domain-based files).
This module re-exports them so that existing code using
``from trulens.feedback import prompts`` keeps working.

Prefer importing from ``trulens.feedback.templates`` in new code.
"""

# --- Agent domain (from templates.agent) --------------------------
from trulens.feedback.templates.agent import ExecutionEfficiency
from trulens.feedback.templates.agent import LogicalConsistency
from trulens.feedback.templates.agent import PlanAdherence
from trulens.feedback.templates.agent import PlanQuality

# --- Generic scaffolding (from templates.base) --------------------
from trulens.feedback.templates.base import COT_REASONS_TEMPLATE
from trulens.feedback.templates.base import LANGCHAIN_PROMPT_TEMPLATE_SYSTEM
from trulens.feedback.templates.base import LANGCHAIN_PROMPT_TEMPLATE_USER
from trulens.feedback.templates.base import (
    LANGCHAIN_PROMPT_TEMPLATE_WITH_COT_REASONS_SYSTEM,
)
from trulens.feedback.templates.base import REMOVE_Y_N

# --- Quality domain (from templates.quality) ----------------------
from trulens.feedback.templates.quality import AGREEMENT_SYSTEM
from trulens.feedback.templates.quality import CORRECT_SYSTEM
from trulens.feedback.templates.quality import Coherence
from trulens.feedback.templates.quality import Conciseness
from trulens.feedback.templates.quality import Controversiality
from trulens.feedback.templates.quality import Correctness
from trulens.feedback.templates.quality import Helpfulness
from trulens.feedback.templates.quality import Sentiment

# --- RAG domain (from templates.rag) ------------------------------
from trulens.feedback.templates.rag import GENERATE_KEY_POINTS_SYSTEM_PROMPT
from trulens.feedback.templates.rag import GENERATE_KEY_POINTS_USER_PROMPT
from trulens.feedback.templates.rag import GROUNDEDNESS_NLI_REASON_FORMAT
from trulens.feedback.templates.rag import GROUNDEDNESS_REASON_TEMPLATE
from trulens.feedback.templates.rag import SYSTEM_FIND_SUPPORTING
from trulens.feedback.templates.rag import USER_FIND_SUPPORTING
from trulens.feedback.templates.rag import Abstention
from trulens.feedback.templates.rag import Answerability
from trulens.feedback.templates.rag import Comprehensiveness
from trulens.feedback.templates.rag import ContextRelevance
from trulens.feedback.templates.rag import Groundedness
from trulens.feedback.templates.rag import PromptResponseRelevance
from trulens.feedback.templates.rag import Trivial

# --- Safety domain (from templates.safety) ------------------------
from trulens.feedback.templates.safety import Criminality
from trulens.feedback.templates.safety import Harmfulness
from trulens.feedback.templates.safety import Insensitivity
from trulens.feedback.templates.safety import Maliciousness
from trulens.feedback.templates.safety import Misogyny
from trulens.feedback.templates.safety import Stereotypes

# --- Re-export aliases matching the old flat variable names -------
LLM_GROUNDEDNESS_SYSTEM = Groundedness.system_prompt
LLM_GROUNDEDNESS_USER = Groundedness.user_prompt
LLM_GROUNDEDNESS_SENTENCES_SPLITTER = Groundedness.sentences_splitter_prompt

LLM_ANSWERABILITY_SYSTEM = Answerability.system_prompt
LLM_ANSWERABILITY_USER = Answerability.user_prompt

LLM_ABSTENTION_SYSTEM = Abstention.system_prompt
LLM_ABSTENTION_USER = Abstention.user_prompt

LLM_TRIVIAL_SYSTEM = Trivial.system_prompt
LLM_TRIVIAL_USER = Trivial.user_prompt

CONTEXT_RELEVANCE_SYSTEM = ContextRelevance.system_prompt
CONTEXT_RELEVANCE_USER = ContextRelevance.user_prompt
CONTEXT_RELEVANCE_DEFAULT_COT_PROMPT = ContextRelevance.default_cot_prompt

ANSWER_RELEVANCE_SYSTEM = PromptResponseRelevance.system_prompt
ANSWER_RELEVANCE_USER = PromptResponseRelevance.user_prompt

SENTIMENT_SYSTEM = Sentiment.system_prompt
SENTIMENT_USER = Sentiment.user_prompt

LANGCHAIN_CONCISENESS_SYSTEM_PROMPT = Conciseness.system_prompt
LANGCHAIN_CORRECTNESS_SYSTEM_PROMPT = Correctness.system_prompt
LANGCHAIN_COHERENCE_SYSTEM_PROMPT = Coherence.system_prompt
LANGCHAIN_HARMFULNESS_SYSTEM_PROMPT = Harmfulness.system_prompt
LANGCHAIN_MALICIOUSNESS_SYSTEM_PROMPT = Maliciousness.system_prompt
LANGCHAIN_HELPFULNESS_SYSTEM_PROMPT = Helpfulness.system_prompt
LANGCHAIN_CONTROVERSIALITY_SYSTEM_PROMPT = Controversiality.system_prompt
LANGCHAIN_MISOGYNY_SYSTEM_PROMPT = Misogyny.system_prompt
LANGCHAIN_CRIMINALITY_SYSTEM_PROMPT = Criminality.system_prompt
LANGCHAIN_INSENSITIVITY_SYSTEM_PROMPT = Insensitivity.system_prompt

STEREOTYPES_SYSTEM_PROMPT = Stereotypes.system_prompt
STEREOTYPES_USER_PROMPT = Stereotypes.user_prompt

COMPREHENSIVENESS_SYSTEM_PROMPT = Comprehensiveness.system_prompt
COMPREHENSIVENESS_USER_PROMPT = Comprehensiveness.user_prompt

LOGICAL_CONSISTENCY_SYSTEM_PROMPT = LogicalConsistency.system_prompt
EXECUTION_EFFICIENCY_SYSTEM_PROMPT = ExecutionEfficiency.system_prompt
PLAN_ADHERENCE_SYSTEM_PROMPT = PlanAdherence.system_prompt
PLAN_QUALITY_SYSTEM_PROMPT = PlanQuality.system_prompt
