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

## Sampling for online evaluation

On high-traffic apps, evaluating every trace is costly. You can
configure sampling so that only a fraction of logged traces are
evaluated automatically, while still logging all traces.

Use `session.configure_online_eval()` to control sampling. A second
call fully replaces the previous configuration.

!!! example "Evaluate 10% of traces"

    ```python
    from trulens.core import TruSession

    session = TruSession()
    session.configure_online_eval(
        sample_rate=0.1,  # evaluate ~10% of logged traces
    )
    ```

### Per-app sampling rates

Pass a dictionary to `sample_rate` to set different rates per app.
Apps not in the dictionary are **not affected** by sampling and
evaluate at 100%.

!!! example "Per-app rates"

    ```python
    session.configure_online_eval(
        sample_rate={
            "prod_rag": 0.1,   # sample 10% for this high-traffic app
            "staging_rag": 1.0, # evaluate everything in staging
        },
    )
    ```

### Throttle and cost budget

You can also limit the rate of evaluations and set a daily cost cap.

- **`throttle`**: Maximum evaluations per minute.
- **`cost_budget`**: Daily USD cap. Resets at UTC midnight. Only
  enforceable for providers that report costs (OpenAI, LiteLLM,
  Google, Cortex). If a provider does not report costs, a warning is
  logged at configuration time.

!!! example "Throttle and budget"

    ```python
    session.configure_online_eval(
        sample_rate=0.1,
        throttle=100,       # max 100 evaluations per minute
        cost_budget=10.0,   # daily $10 cap
    )
    ```

Sampling decisions are deterministic: the same `record_id` always
produces the same decision, so results are reproducible across retries
and processes.

### Inspecting coverage

Evaluated results carry sampling metadata. After calling
`get_records_and_feedback()`, the returned DataFrame includes:

- **`sampled`**: `True` if the record was evaluated, `False` if
  skipped, `None` if sampling was not configured.
- **`sample_rate`**: The rate that was active when the decision was
  made.
- **`eval_decision_reason`**: Why the record was or was not evaluated
  (`evaluated`, `not_sampled`, `throttled`, `over_budget`).

!!! example "Filtering by sampling status"

    ```python
    records, feedback_cols = session.get_records_and_feedback(
        app_name="prod_rag",
    )

    evaluated = records[records["sampled"] == True]
    skipped = records[records["sampled"] == False]
    ```

### Backfilling skipped records

Explicit `compute_now()` calls are **never gated** by sampling.
Records that were skipped by the automatic evaluator can always be
backfilled later:

!!! example "Backfill skipped records"

    ```python
    skipped_ids = records[records["sampled"] == False]["record_id"].tolist()
    app._evaluator.compute_now(record_ids=skipped_ids)
    ```

### Monitoring sampling decisions

The sampling controller exposes counters for each decision reason:

!!! example "Check counters"

    ```python
    counters = session.sampling_controller.counters
    # {'evaluated': 50, 'not_sampled': 450, 'throttled': 0, 'over_budget': 0, ...}
    ```
