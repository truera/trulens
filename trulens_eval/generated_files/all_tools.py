#!/usr/bin/env python
# coding: utf-8

# # Quickstart
#
# In this quickstart you will create a simple LLM Chain and learn how to log it and get feedback on an LLM response.

# ## Setup
# ### Add API keys
# For this quickstart you will need Open AI and Huggingface keys

import os

os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."

# ### Import from LangChain and TruLens

# Imports main tools:
from trulens_eval import TruChain, Feedback, Huggingface, Tru, Query

tru = Tru()

# Imports from langchain to build app. You may need to install langchain first
# with the following:
# ! pip install langchain>=0.0.170
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

# ### Create Simple LLM Application
#
# This example uses a LangChain framework and OpenAI LLM

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

prompt_input = '¿que hora es?'

llm_response = chain(prompt_input)

print(llm_response)

# ## Initialize Feedback Function(s)

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on(
    text1=Query.RecordInput, text2=Query.RecordOutput
)

# ## Instrument chain for logging with TruLens

truchain = TruChain(
    chain, app_id='Chain3_ChatApplication', feedbacks=[f_lang_match]
)

# Instrumented chain can operate like the original:
llm_response = truchain(prompt_input)

print(llm_response)

# ## Explore in a Dashboard

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# ### Chain Leaderboard
#
# Understand how your LLM application is performing at a glance. Once you've set up logging and evaluation in your application, you can view key performance statistics including cost and average feedback value across all of your LLM apps using the chain leaderboard. As you iterate new versions of your LLM application, you can compare their performance across all of the different quality metrics you've set up.
#
# Note: Average feedback values are returned and printed in a range from 0 (worst) to 1 (best).
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
# Click on a record to dive deep into all of the details of your chain stack and underlying LLM, captured by tru_chain.
#
# ![Explore a Chain](https://www.trulens.org/Assets/image/Chain_Explore.png)
#
# If you prefer the raw format, you can quickly get it using the "Display full chain json" or "Display full record json" buttons at the bottom of the page.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.

# ## Or view results directly in your notebook

tru.get_records_and_feedback(app_ids=[]
                            )[0]  # pass an empty list of app_ids to get all

# # Logging
#
# ## Automatic Logging
#
# The simplest method for logging with TruLens is by wrapping with TruChain and including the tru argument, as shown in the quickstart.
#
# This is done like so:

truchain = TruChain(chain, app_id='Chain1_ChatApplication', tru=tru)
truchain("This will be automatically logged.")

# Feedback functions can also be logged automatically by providing them in a list to the feedbacks arg.

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

tc = TruChain(chain, app_id='Chain1_ChatApplication')

# ### Set up logging and instrumentation
#
# Making the first call to your wrapped LLM Application will now also produce a log or "record" of the chain execution.
#

prompt_input = 'que hora es?'
gpt3_response, record = tc.call_with_record(prompt_input)

# We can log the records but first we need to log the chain itself.

tru.add_app(app=truchain)

# Then we can log the record:

tru.add_record(record)

# ### Log App Feedback
# Capturing app feedback such as user feedback of the responses can be added with one call.

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

feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[f_lang_match]
)
print(feedback_results)

# After capturing feedback, you can then log it to your local database.

tru.add_feedbacks(feedback_results)

# ### Out-of-band Feedback evaluation
#
# In the above example, the feedback function evaluation is done in the same process as the chain evaluation. The alternative approach is the use the provided persistent evaluator started via `tru.start_deferred_feedback_evaluator`. Then specify the `feedback_mode` for `TruChain` as `deferred` to let the evaluator handle the feedback functions.
#
# For demonstration purposes, we start the evaluator here but it can be started in another process.

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

# # Out-of-the-box Feedback Functions
# See: <https://www.trulens.org/trulens_eval/api/tru_feedback/>
#
# ## Relevance
#
# This evaluates the *relevance* of the LLM response to the given text by LLM prompting.
#
# Relevance is currently only available with OpenAI ChatCompletion API.
#
# ## Sentiment
#
# This evaluates the *positive sentiment* of either the prompt or response.
#
# Sentiment is currently available to use with OpenAI, HuggingFace or Cohere as the model provider.
#
# * The OpenAI sentiment feedback function prompts a Chat Completion model to rate the sentiment from 1 to 10, and then scales the response down to 0-1.
# * The HuggingFace sentiment feedback function returns a raw score from 0 to 1.
# * The Cohere sentiment feedback function uses the classification endpoint and a small set of examples stored in `feedback_prompts.py` to return either a 0 or a 1.
#
# ## Model Agreement
#
# Model agreement uses OpenAI to attempt an honest answer at your prompt with system prompts for correctness, and then evaluates the agreement of your LLM response to this model on a scale from 1 to 10. The agreement with each honest bot is then averaged and scaled from 0 to 1.
#
# ## Language Match
#
# This evaluates if the language of the prompt and response match.
#
# Language match is currently only available to use with HuggingFace as the model provider. This feedback function returns a score in the range from 0 to 1, where 1 indicates match and 0 indicates mismatch.
#
# ## Toxicity
#
# This evaluates the toxicity of the prompt or response.
#
# Toxicity is currently only available to be used with HuggingFace, and uses a classification endpoint to return a score from 0 to 1. The feedback function is negated as not_toxicity, and returns a 1 if not toxic and a 0 if toxic.
#
# ## Moderation
#
# The OpenAI Moderation API is made available for use as feedback functions. This includes hate, hate/threatening, self-harm, sexual, sexual/minors, violence, and violence/graphic. Each is negated (ex: not_hate) so that a 0 would indicate that the moderation rule is violated. These feedback functions return a score in the range 0 to 1.
#
# # Adding new feedback functions
#
# Feedback functions are an extensible framework for evaluating LLMs. You can add your own feedback functions to evaluate the qualities required by your application by updating `trulens_eval/tru_feedback.py`. If your contributions would be useful for others, we encourage you to contribute to TruLens!
#
# Feedback functions are organized by model provider into Provider classes.
#
# The process for adding new feedback functions is:
# 1. Create a new Provider class or locate an existing one that applies to your feedback function. If your feedback function does not rely on a model provider, you can create a standalone class. Add the new feedback function method to your selected class. Your new method can either take a single text (str) as a parameter or both prompt (str) and response (str). It should return a float between 0 (worst) and 1 (best).

from trulens_eval import Provider


class StandAlone(Provider):

    def my_custom_feedback(self, my_text_field: str) -> float:
        """
        A dummy function of text inputs to float outputs.

        Parameters:
            my_text_field (str): Text to evaluate.

        Returns:
            float: square length of the text
        """
        return 1.0 / (1.0 + len(my_text_field) * len(my_text_field))


# 2. Instantiate your provider and feedback functions. The feedback function is wrapped by the trulens-eval Feedback class which helps specify what will get sent to your function parameters (For example: Query.RecordInput or Query.RecordOutput)

my_standalone = StandAlone()
my_feedback_function_standalone = Feedback(
    my_standalone.my_custom_feedback
).on(my_text_field=Query.RecordOutput)

# 3. Your feedback function is now ready to use just like the out of the box feedback functions. Below is an example of it being used.

feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[my_feedback_function_standalone]
)
tru.add_feedbacks(feedback_results)
