The primary method for evaluating LLM apps is by running feedback functions with
your app.

To do so, you first need to define the wrap the specified feedback
implementation with `Feedback` and select what components of your app to
evaluate. Optionally, you can also select an aggregation method.

```python
f_context_relevance = Feedback(openai.context_relevance)
    .on_input()
    .on(context)
    .aggregate(numpy.min)

# Implementation signature:
# def context_relevance(self, question: str, statement: str) -> float:
```

Once you've defined the feedback functions to run with your application, you can
then pass them as a list to the instrumentation class of your choice, along with
the app itself. These make up the `recorder`.

```python
from trulens_eval import TruChain
# f_lang_match, f_qa_relevance, f_context_relevance are feedback functions
tru_recorder = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match, f_qa_relevance, f_context_relevance])
```

Now that you've included the evaluations as a component of your `recorder`, they
are able to be run with your application. By default, feedback functions will be
run in the same process as the app. This is known as the feedback mode:
`with_app_thread`.

```python
with tru_recorder as recording:
    chain(""What is langchain?")
```

In addition to `with_app_thread`, there are a number of other manners of running
feedback functions. These are accessed by the feedback mode and included when
you construct the recorder, like so:

```python
from trulens_eval import FeedbackMode

tru_recorder = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match, f_qa_relevance, f_context_relevance],
    feedback_mode=FeedbackMode.DEFERRED
    )
```

Here are the different feedback modes you can use:

- `WITH_APP_THREAD`: This is the default mode. Feedback functions will run in the
  same process as the app, but only after the app has produced a record.
- `NONE`: In this mode, no evaluation will occur, even if feedback functions are
  specified.
- `WITH_APP`: Feedback functions will run immediately and before the app returns a
  record.
- `DEFERRED`: Feedback functions will be evaluated later via the process started
  by `tru.start_evaluator`.
