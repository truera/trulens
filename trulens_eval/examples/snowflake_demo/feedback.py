import os

from custom_feedback.small_local_models import SmallLocalModels
import numpy as np
import streamlit as st

from trulens_eval import Feedback
from trulens_eval import Select
from trulens_eval import Tru
from trulens_eval.feedback.provider.cortex import Cortex
from trulens_eval.feedback.provider.litellm import LiteLLM

db_url = "snowflake://{user}:{password}@{account}/{dbname}/{schema}?warehouse={warehouse}&role={role}".format(
    user=os.environ["SNOWFLAKE_USER"],
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    password=os.environ["SNOWFLAKE_USER_PASSWORD"],
    dbname=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    role=os.environ["SNOWFLAKE_ROLE"],
)

tru = Tru(database_url=db_url)

AVAILABLE_PROVIDERS = ["Replicate", "Cortex"]


@st.cache_resource
def get_provider(provider_name: str):
    if provider_name == "Replicate":
        return LiteLLM(
            model_engine="replicate/snowflake/snowflake-arctic-instruct"
        )
    elif provider_name == "Cortex":
        return Cortex(model_engine="snowflake-arctic")
    elif provider_name in AVAILABLE_PROVIDERS:
        raise NotImplementedError(
            f"Provider {provider_name} is not yet implemented."
        )
    else:
        raise ValueError("Invalid provider name", provider_name)


@st.cache_resource
def get_feedbacks(provider_name: str, use_rag: bool = True):
    provider = get_provider(provider_name)
    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        ).on(Select.RecordCalls.retrieve_context.rets[:].collect()).on_output()
    )
    f_context_relevance = (
        Feedback(provider.context_relevance,
                 name="Context Relevance").on_input().on(
                     Select.RecordCalls.retrieve_context.rets[:]
                 ).aggregate(
                     np.mean
                 )  # choose a different aggregation method if you wish
    )
    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons,
                 name="Answer Relevance").on_input().on_output().aggregate(
                     np.mean
                 )
    )
    if use_rag:
        return [
            f_context_relevance,
            #f_small_local_models_context_relevance,
            f_answer_relevance,
            f_groundedness,
        ]
    else:
        return [f_answer_relevance]


provider = get_provider("Cortex")

f_context_relevance = (
    Feedback(provider.context_relevance,
             name="Context Relevance").on_input().on(
                 Select.RecordCalls.retrieve_context.rets[:]
             ).aggregate(np.mean
                        )  # choose a different aggregation method if you wish
)

small_local_model_provider = SmallLocalModels()
f_small_local_models_context_relevance = (
    Feedback(
        small_local_model_provider.context_relevance,
        name="[Small Local Model] Context Relevance",
    ).on_input().on(Select.RecordCalls.retrieve_context.rets[:]).aggregate(
        np.mean
    )  # choose a different aggregation method if you wish
)
