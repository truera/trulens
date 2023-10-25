#!/usr/bin/env python
# coding: utf-8

# # Langchain Quickstart
#
# In this quickstart you will create a simple LLM Chain and learn how to log it and get feedback on an LLM response.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/langchain_quickstart.ipynb)

# In[ ]:

# ! pip install trulens_eval==0.16.0 langchain>=0.0.263

# ## Setup
# ### Add API keys
# For this quickstart you will need Open AI and Huggingface keys

# In[ ]:

import os

os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."

# ### Import from LangChain and TruLens

# In[ ]:

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import Huggingface
from trulens_eval import Tru
from trulens_eval.schema import FeedbackResult

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
# This example uses a LangChain framework and OpenAI LLM.

# In[ ]:


def new_session() -> LLMChain:
    # A function to return a chain for a new session. This is needed if you
    # would like to run your app from the dashboard.

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

    return chain


chain = new_session()

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

tru_recorder = tru.Chain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    new_session=new_session  # to interact with app in the dashboard
)

# In[ ]:

with tru_recorder as recording:
    llm_response = chain(prompt_input)

display(llm_response)

# ## Retrieve records and feedback

# In[ ]:

# The record of the ap invocation can be retrieved from the `recording`:

rec = recording.get()  # use .get if only one record
# recs = recording.records # use .records if multiple

display(rec)

# In[ ]:

# The results of the feedback functions can be rertireved from the record. These
# are `Future` instances (see `concurrent.futures`). You can use `as_completed`
# to wait until they have finished evaluating.

from concurrent.futures import as_completed

for feedback_future in as_completed(rec.feedback_results):
    feedback, feedback_result = feedback_future.result()

    feedback: Feedback
    feedbac_result: FeedbackResult

    display(feedback.name, feedback_result.result)

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
# ![Chain Leaderboard](https://www.trulens.org/assets/images/Leaderboard.png)
#
# To dive deeper on a particular chain, click "Select Chain".
#
# ### Understand chain performance with Evaluations
#
# To learn more about the performance of a particular chain or LLM model, we can select it to view its evaluations at the record level. LLM quality is assessed through the use of feedback functions. Feedback functions are extensible methods for determining the quality of LLM responses and can be applied to any downstream LLM task. Out of the box we provide a number of feedback functions for assessing model agreement, sentiment, relevance and more.
#
# The evaluations tab provides record-level metadata and feedback on the quality of your LLM application.
#
# ![Evaluations](https://www.trulens.org/assets/images/Leaderboard.png)
#
# ### Deep dive into full chain metadata
#
# Click on a record to dive deep into all of the details of your chain stack and underlying LLM, captured by tru_chain_recorder.
#
# ![Explore a Chain](https://www.trulens.org/assets/images/Chain_Explore.png)
#
# If you prefer the raw format, you can quickly get it using the "Display full chain json" or "Display full record json" buttons at the bottom of the page.
#
# ### Run the Chain from the Dashboard
#
# ![App Runner](https://www.trulens.org/assets/images/appui/running_session.png)
#
# You can run the chain inside the dashboard by creating a new session on the "Apps" page. See more information about this feature in `dashboard_appui.ipynb`.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.

# ## Or view results directly in your notebook

# In[ ]:

tru.get_records_and_feedback(app_ids=[]
                            )[0]  # pass an empty list of app_ids to get all
