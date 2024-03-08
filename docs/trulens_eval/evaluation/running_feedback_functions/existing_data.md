In many cases, developers have already logged runs of an LLM app they wish to evaluate or wish to log their app using another system. Feedback functions can also be run on existing data, independent of the `recorder`.

At the most basic level, feedback implementations are simple callables that can be run on any arguments
matching their signatures like so:

```python
feedback_result = provider.relevance("<some prompt>", "<some response>")
```

!!! note

    Running the feedback implementation in isolation will not log the evaluation results in TruLens.

In the case that you have already logged a run of your application with TruLens and have the record available, the process for running an (additional) evaluation on that record is by using `tru.run_feedback_functions`:

```python
tru_rag = TruCustomApp(rag, app_id = 'RAG v1')

result, record = tru_rag.with_record(rag.query, "How many professors are at UW in Seattle?")
feedback_results = tru.run_feedback_functions(record, feedbacks=[f_lang_match, f_qa_relevance, f_context_relevance])
tru.add_feedbacks(feedback_results)
```

### TruVirtual

If your application was run (and logged) outside of TruLens, `TruVirtual` can be used to ingest and evaluate the logs.

The first step to loading your app logs into TruLens is creating a virtual app. This virtual app can be a plain dictionary or use our `VirtualApp` class to store any information you would like. You can refer to these values for evaluating feedback.

```python
virtual_app = dict(
    llm=dict(
        modelname="some llm component model name"
    ),
    template="information about the template I used in my app",
    debug="all of these fields are completely optional"
)
from trulens_eval import Select
from trulens_eval.tru_virtual import VirtualApp

virtual_app = VirtualApp(virtual_app) # can start with the prior dictionary
virtual_app[Select.RecordCalls.llm.maxtokens] = 1024
```

When setting up the virtual app, you should also include any components that you would like to evaluate in the virtual app. This can be done using the `Select` class. Using selectors here lets use reuse the setup you use to define feedback functions. Below you can see how to set up a virtual app with a retriever component, which will be used later in the example for feedback evaluation.

```python
from trulens_eval import Select
retriever_component = Select.RecordCalls.retriever
virtual_app[retriever_component] = "this is the retriever component"
```

Now that you've set up your virtual app, you can use it to store your logged data.

To incorporate your data into TruLens, you have two options. You can either create a `Record` directly, or you can use the `VirtualRecord` class, which is designed to help you build records so they can be ingested to TruLens.

The parameters you'll use with `VirtualRecord` are the same as those for `Record`, with one key difference: calls are specified using selectors.

In the example below, we add two records. Each record includes the inputs and outputs for a context retrieval component. Remember, you only need to provide the information that you want to track or evaluate. The selectors are references to methods that can be selected for feedback, as we'll demonstrate below.

```python
from trulens_eval.tru_virtual import VirtualRecord

# The selector for a presumed context retrieval component's call to
# `get_context`. The names are arbitrary but may be useful for readability on
# your end.
context_call = retriever_component.get_context

rec1 = VirtualRecord(
    main_input="Where is Germany?",
    main_output="Germany is in Europe",
    calls=
        {
            context_call: dict(
                args=["Where is Germany?"],
                rets=["Germany is a country located in Europe."]
            )
        }
    )
rec2 = VirtualRecord(
    main_input="Where is Germany?",
    main_output="Poland is in Europe",
    calls=
        {
            context_call: dict(
                args=["Where is Germany?"],
                rets=["Poland is a country located in Europe."]
            )
        }
    )

data = [rec1, rec2]
```

Alternatively, suppose we have an existing dataframe of prompts, contexts and responses we wish to ingest.

```python
import pandas as pd

data = {
    'prompt': ['Where is Germany?', 'What is the capital of France?'],
    'response': ['Germany is in Europe', 'The capital of France is Paris'],
    'context': ['Germany is a country located in Europe.', 'France is a country in Europe and its capital is Paris.']
}
df = pd.DataFrame(data)
df.head()
```

To ingest the data in this form, we can iterate through the dataframe to ingest each prompt, context and response into virtual records.

```python
data_dict = df.to_dict('records')

data = []

for record in data_dict:
    rec = VirtualRecord(
        main_input=record['prompt'],
        main_output=record['response'],
        calls=
            {
                context_call: dict(
                    args=[record['prompt']],
                    rets=[record['context']]
                )
            }
        )
    data.append(rec)
```

Now that we've ingested constructed the virtual records, we can build our feedback functions. This is done just the same as normal, except the context selector will instead refer to the new `context_call` we added to the virtual record.

```python
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.feedback.feedback import Feedback

# Initialize provider class
openai = OpenAI()

# Select context to be used in feedback. We select the return values of the
# virtual `get_context` call in the virtual `retriever` component. Names are
# arbitrary except for `rets`.
context = context_call.rets[:]

# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(openai.qs_relevance)
    .on_input()
    .on(context)
)
```

Then, the feedback functions can be passed to `TruVirtual` to construct the `recorder`. Most of the fields that other non-virtual apps take can also be specified here.

```python
from trulens_eval.tru_virtual import TruVirtual

virtual_recorder = TruVirtual(
    app_id="a virtual app",
    app=virtual_app,
    feedbacks=[f_context_relevance]
)
```

To finally ingest the record and run feedbacks, we can use `add_record`.

```python
for record in data:
    virtual_recorder.add_record(rec)
```

To optionally store metadata about your application, you can also pass an arbitrary `dict` to `VirtualApp`. This information can also be used in evaluation.

```python
virtual_app = dict(
    llm=dict(
        modelname="some llm component model name"
    ),
    template="information about the template I used in my app",
    debug="all of these fields are completely optional"
)

from trulens_eval.schema import Select
from trulens_eval.tru_virtual import VirtualApp

virtual_app = VirtualApp(virtual_app)
```

The `VirtualApp` metadata can also be appended.

```python
virtual_app[Select.RecordCalls.llm.maxtokens] = 1024
```

This can be particularly useful for storing the components of an LLM app to be later used for evaluation.

```python
retriever_component = Select.RecordCalls.retriever
virtual_app[retriever_component] = "this is the retriever component"
```
