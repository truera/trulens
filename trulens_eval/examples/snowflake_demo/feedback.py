from trulens_eval import Tru
from trulens_eval import Feedback, Select
from trulens_eval.feedback.provider.litellm import LiteLLM
import numpy as np

db_url = 'snowflake://{user}:{password}@{account}/{dbname}/{schema}?warehouse={warehouse}&role={role}'.format(
    user='<user>',
    account='<account>',
    password='<password>',
    dbname='<db>',
    schema='<schema>',
    warehouse='<wh>',
    role="<role>"
)

tru = Tru(database_url=db_url)

provider = LiteLLM(model_engine="replicate/snowflake/snowflake-arctic-instruct")

f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve_context.rets[1][:])
    .on_output()
)

f_context_relevance = (
    Feedback(provider.context_relevance, name = "Context Relevance")
    .on_input()
    .on(Select.RecordCalls.retrieve_context.rets[1][:])
    .aggregate(np.mean) # choose a different aggregation method if you wish
)
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name = "Answer Relevance")
    .on_input()
    .on_output()
    .aggregate(np.mean)
)
f_criminality_input = (
    Feedback(provider.criminality_with_cot_reasons,
             name = "Criminality input",
             higher_is_better=False)
             .on(Select.RecordInput)
)
f_criminality_output = (
    Feedback(provider.criminality_with_cot_reasons,
             name = "Criminality output",
             higher_is_better=False)
             .on_output()
)

feedbacks_rag = [f_context_relevance, f_answer_relevance, f_groundedness, f_criminality_input, f_criminality_output]
feedbacks_no_rag = [f_answer_relevance, f_criminality_input, f_criminality_output]
