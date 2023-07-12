```python
from trulens_eval import Tru
from trulens_eval import TruChain

tru = Tru()
```

This example uses LangChain and OpenAI, but the same process can be followed with any framework and model provider.


```python
# imports from LangChain to build app
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

import os
os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."
```


```python
# create LLM chain
full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="Provide a helpful response with relevant background information for the following: {prompt}",
            input_variables=["prompt"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)

chain = LLMChain(llm=chat, prompt=chat_prompt_template)
```

Now that we created an LLM chain, we can set up our first feedback function. Here, we'll create a feedback function for language matching. After we've created the feedback function, we can include it in the TruChain wrapper. Now, whenever our wrapped chain is used we'll log both the metadata and feedback.


```python
# create a feedback function

from trulens_eval.feedback import Feedback, Huggingface
```


```python
# Initialize HuggingFace-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

# wrap your chain with TruChain
truchain = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match]
)
# Note: any `feedbacks` specified here will be evaluated and logged whenever the chain is used.
truchain("que hora es?")
```

Now you can explore your LLM-based application!

Doing so will help you understand how your LLM application is performing at a glance. As you iterate new versions of your LLM application, you can compare their performance across all of the different quality metrics you've set up. You'll also be able to view evaluations at a record level, and explore the chain metadata for each record.


```python
tru.run_dashboard() # open a Streamlit app to explore
```

Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

For more information, see [TruLens-Eval Documentation](https://www.trulens.org/trulens_eval/quickstart/).


