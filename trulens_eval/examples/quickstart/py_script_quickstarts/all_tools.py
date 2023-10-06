#!/usr/bin/env python
# coding: utf-8

# # Langchain Quickstart
#
# In this quickstart you will create a simple LLM Chain and learn how to log it and get feedback on an LLM response.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/langchain_quickstart.ipynb)

# In[ ]:

# ! pip install trulens_eval==0.15.1 langchain>=0.0.263

# ## Setup
# ### Add API keys
# For this quickstart you will need Open AI and Huggingface keys

# In[ ]:

import os

os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."

# ### Import from LangChain and TruLens

# In[ ]:

from IPython.display import JSON

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import Huggingface
from trulens_eval import Tru
from trulens_eval import TruChain

tru = Tru()

# Imports from langchain to build app. You may need to install langchain first
# with the following:
# ! pip install langchain>=0.0.170
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import PromptTemplate

# ### Create Simple LLM Application
#
# This example uses a LangChain framework and OpenAI LLM

# In[ ]:

full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=
        "Provide a helpful response with relevant background information for the following: {prompt}",
        input_variables=["prompt"],
    )
)

chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

llm = OpenAI(temperature=0.9, max_tokens=128)

chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)

# ### Send your first request

# In[ ]:

prompt_input = '¿que hora es?'

# In[ ]:

llm_response = chain(prompt_input)

display(llm_response)

# ## Initialize Feedback Function(s)

# In[ ]:

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

# ## Instrument chain for logging with TruLens

# In[ ]:

tru_recorder = TruChain(
    chain, app_id='Chain1_ChatApplication', feedbacks=[f_lang_match]
)

# In[ ]:

with tru_recorder as recording:
    llm_response = chain(prompt_input)

display(llm_response)

# ## Explore in a Dashboard

# In[ ]:

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# ### Chain Leaderboard
#
# Understand how your LLM application is performing at a glance. Once you've set up logging and evaluation in your application, you can view key performance statistics including cost and average feedback value across all of your LLM apps using the chain leaderboard. As you iterate new versions of your LLM application, you can compare their performance across all of the different quality metrics you've set up.
#
# Note: Average feedback values are returned and displayed in a range from 0 (worst) to 1 (best).
#
# ![Chain Leaderboard](https://www.trulens.org/Assets/image/Leaderboard.png)
#
# To dive deeper on a particular chain, click "Select Chain".
#
# ### Understand chain performance with Evaluations
#
# To learn more about the performance of a particular chain or LLM model, we can select it to view its evaluations at the record level. LLM quality is assessed through the use of feedback functions. Feedback functions are extensible methods for determining the quality of LLM responses and can be applied to any downstream LLM task. Out of the box we provide a number of feedback functions for assessing model agreement, sentiment, relevance and more.
#
# The evaluations tab provides record-level metadata and feedback on the quality of your LLM application.
#
# ![Evaluations](https://www.trulens.org/Assets/image/Leaderboard.png)
#
# ### Deep dive into full chain metadata
#
# Click on a record to dive deep into all of the details of your chain stack and underlying LLM, captured by tru_chain_recorder.
#
# ![Explore a Chain](https://www.trulens.org/Assets/image/Chain_Explore.png)
#
# If you prefer the raw format, you can quickly get it using the "Display full chain json" or "Display full record json" buttons at the bottom of the page.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.

# ## Or view results directly in your notebook

# In[ ]:

tru.get_records_and_feedback(app_ids=[]
                            )[0]  # pass an empty list of app_ids to get all

# # Logging Methods
#
# ## Automatic Logging
#
# The simplest method for logging with TruLens is by wrapping with TruChain and including the tru argument, as shown in the quickstart.
#
# This is done like so:

# In[ ]:

truchain = TruChain(chain, app_id='Chain1_ChatApplication', tru=tru)
truchain("This will be automatically logged.")

# Feedback functions can also be logged automatically by providing them in a list to the feedbacks arg.

# In[ ]:

truchain = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],  # feedback functions
    tru=tru
)
truchain("This will be automatically logged.")

# ## Manual Logging
#
# ### Wrap with TruChain to instrument your chain

# In[ ]:

tc = TruChain(chain, app_id='Chain1_ChatApplication')

# ### Set up logging and instrumentation
#
# Making the first call to your wrapped LLM Application will now also produce a log or "record" of the chain execution.
#

# In[ ]:

prompt_input = 'que hora es?'
gpt3_response, record = tc.call_with_record(prompt_input)

# We can log the records but first we need to log the chain itself.

# In[ ]:

tru.add_app(app=truchain)

# Then we can log the record:

# In[ ]:

tru.add_record(record)

# ### Log App Feedback
# Capturing app feedback such as user feedback of the responses can be added with one call.

# In[ ]:

thumb_result = True
tru.add_feedback(
    name="👍 (1) or 👎 (0)", record_id=record.record_id, result=thumb_result
)

# ### Evaluate Quality
#
# Following the request to your app, you can then evaluate LLM quality using feedback functions. This is completed in a sequential call to minimize latency for your application, and evaluations will also be logged to your local machine.
#
# To get feedback on the quality of your LLM, you can use any of the provided feedback functions or add your own.
#
# To assess your LLM quality, you can provide the feedback functions to `tru.run_feedback()` in a list provided to `feedback_functions`.
#

# In[ ]:

feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[f_lang_match]
)
display(feedback_results)

# After capturing feedback, you can then log it to your local database.

# In[ ]:

tru.add_feedbacks(feedback_results)

# ### Out-of-band Feedback evaluation
#
# In the above example, the feedback function evaluation is done in the same process as the chain evaluation. The alternative approach is the use the provided persistent evaluator started via `tru.start_deferred_feedback_evaluator`. Then specify the `feedback_mode` for `TruChain` as `deferred` to let the evaluator handle the feedback functions.
#
# For demonstration purposes, we start the evaluator here but it can be started in another process.

# In[ ]:

truchain: TruChain = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    tru=tru,
    feedback_mode="deferred"
)

tru.start_evaluator()
truchain("This will be logged by deferred evaluator.")
tru.stop_evaluator()

# # Custom Functions
#
# Feedback functions are an extensible framework for evaluating LLMs. You can add your own feedback functions to evaluate the qualities required by your application by updating `trulens_eval/feedback.py`, or simply creating a new provider class and feedback function in youre notebook. If your contributions would be useful for others, we encourage you to contribute to TruLens!
#
# Feedback functions are organized by model provider into Provider classes.
#
# The process for adding new feedback functions is:
# 1. Create a new Provider class or locate an existing one that applies to your feedback function. If your feedback function does not rely on a model provider, you can create a standalone class. Add the new feedback function method to your selected class. Your new method can either take a single text (str) as a parameter or both prompt (str) and response (str). It should return a float between 0 (worst) and 1 (best).

# In[ ]:

from trulens_eval import Feedback
from trulens_eval import Provider
from trulens_eval import Select
from trulens_eval import Tru


class StandAlone(Provider):

    def custom_feedback(self, my_text_field: str) -> float:
        """
        A dummy function of text inputs to float outputs.

        Parameters:
            my_text_field (str): Text to evaluate.

        Returns:
            float: square length of the text
        """
        return 1.0 / (1.0 + len(my_text_field) * len(my_text_field))


# 2. Instantiate your provider and feedback functions. The feedback function is wrapped by the trulens-eval Feedback class which helps specify what will get sent to your function parameters (For example: Select.RecordInput or Select.RecordOutput)

# In[ ]:

standalone = StandAlone()
f_custom_function = Feedback(standalone.custom_feedback
                            ).on(my_text_field=Select.RecordOutput)

# 3. Your feedback function is now ready to use just like the out of the box feedback functions. Below is an example of it being used.

# In[ ]:

tru = Tru()
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[f_custom_function]
)
tru.add_feedbacks(feedback_results)

# ## Multi-Output Feedback functions
# Trulens also supports multi-output feedback functions. As a typical feedback function will output a float between 0 and 1, multi-output should output a dictionary of `output_key` to a float between 0 and 1. The feedbacks table will display the feedback with column `feedback_name:::outputkey`

# In[ ]:

multi_output_feedback = Feedback(
    lambda input_param: {
        'output_key1': 0.1,
        'output_key2': 0.9
    }, name="multi"
).on(input_param=Select.RecordOutput)
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[multi_output_feedback]
)
tru.add_feedbacks(feedback_results)

# In[ ]:

# Aggregators will run on the same dict keys.
import numpy as np

multi_output_feedback = Feedback(
    lambda input_param: {
        'output_key1': 0.1,
        'output_key2': 0.9
    },
    name="multi-agg"
).on(input_param=Select.RecordOutput).aggregate(np.mean)
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[multi_output_feedback]
)
tru.add_feedbacks(feedback_results)

# In[ ]:


# For multi-context chunking, an aggregator can operate on a list of multi output dictionaries.
def dict_aggregator(list_dict_input):
    agg = 0
    for dict_input in list_dict_input:
        agg += dict_input['output_key1']
    return agg


multi_output_feedback = Feedback(
    lambda input_param: {
        'output_key1': 0.1,
        'output_key2': 0.9
    },
    name="multi-agg-dict"
).on(input_param=Select.RecordOutput).aggregate(dict_aggregator)
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[multi_output_feedback]
)
tru.add_feedbacks(feedback_results)
