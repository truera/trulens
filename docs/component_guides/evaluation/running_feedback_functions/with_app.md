The primary method for evaluating LLM apps is by running metrics with
your app.

To do so, you first need to define the metric by wrapping a metric
implementation with `Metric` and specifying selectors that define what components
of your app to evaluate. Optionally, you can also specify an aggregation method.

!!! example

    ```python
    from trulens.core import Metric, Selector
    import numpy as np

    f_context_relevance = Metric(
        implementation=openai.context_relevance,
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )

    # Implementation signature:
    # def context_relevance(self, question: str, context: str) -> float:
    ```

Once you've defined the metrics to run with your application, you can
then pass them as a list to the instrumentation class of your choice, along with
the app itself. These make up the `recorder`.

!!! example

    ```python
    from trulens.apps.langchain import TruChain

    # f_lang_match, f_qa_relevance, f_context_relevance are metrics
    tru_recorder = TruChain(
        chain,
        app_name='ChatApplication',
        app_version="Chain1",
        feedbacks=[f_lang_match, f_qa_relevance, f_context_relevance],
    )
    ```

Now that you've included the evaluations as a component of your `recorder`, they
are able to be run with your application. By default, metrics will be
run in the same process as the app. This is known as the feedback mode:
`WITH_APP_THREAD`.

!!! example

    ```python
    with tru_recorder as recording:
        chain("What is langchain?")
    ```

In addition to `WITH_APP_THREAD`, there are a number of other manners of running
metrics. These are accessed by the feedback mode and included when
you construct the recorder.

!!! example

    ```python
    from trulens.core import FeedbackMode

    tru_recorder = TruChain(
        chain,
        app_name='ChatApplication',
        app_version="Chain1",
        feedbacks=[f_lang_match, f_qa_relevance, f_context_relevance],
        feedback_mode=FeedbackMode.DEFERRED,
    )
    ```

Here are the different feedback modes you can use:

- `WITH_APP_THREAD`: This is the default mode. Metrics will run in the
  same process as the app, but only after the app has produced a record.
- `NONE`: In this mode, no evaluation will occur, even if metrics are
  specified.
- `WITH_APP`: Metrics will run immediately and before the app returns a
  record.
- `DEFERRED`: Metrics will be evaluated later via the process started
  by `tru.start_evaluator`.
