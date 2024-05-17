# 🎨 Design Goals

Public API is inspired by two main goals discussed in this section. For internal
designs, see also [internals](/trulens_eval/contributing/internals).

## 🎯 Primary Goal: Minimal time/effort to value

If a user already has an LLM app coded in one of the supported libraries, give
them value with the minimal efford beyond that app.

Currently to get going, a user needs to add  3 lines of python:

```python
app = ... # some existing langchain app

from trulens_eval import Tru, TruChain # line 1

with TruChain(app): # line 2
    app.invoke("some question") # doesn't count since they already had this

Tru().start_dashboard() # line 3
```

From here they can open the dashboard and inspect the recording of their app's
invocation including performance and cost statistics. Achieving this goal is the
motivation for the [internal designs](internals) and is
responsible for most of the [tech-debt](techdebt)
incurred by the library.

## 🗺 Secondary Goal: Incremental paths towards further value

Paths include:

### 🚔 Evals

  - Evals via feedback functions. We include a [wide variety of feedback
    functions](/trulens_eval/evaluation/feedback_implementations/stock) for
    typical LLM app components.

  - Automatically evaluate feedback functions. Initializing an
    [App][trulens_eval.app.App] with a set of feedback functions makes them
    evaluate as the wrapped app is used. See also [Running with your
    app](/trulens_eval/evaluation/running_feedback_functions/with_app).

  - Evaluate prior executions of app using [virtual
    apps][trulens_eval.tru_virtual.TruVirtual]. See also [Running on existing
    data](/trulens_eval/evaluation/running_feedback_functions/existing_data).

  - Customize feedback functions. Define [new feedback
    functions](/trulens_eval/evaluation/feedback_implementations/custom_feedback_functions)
    and have them evaluated (automatically).

### 🛰 Scalability

  See also [Moving apps from dev to prod](/trulens_eval/guides/use_cases_production).
  
  - Handle various database types including remote ones via [database
    configuration](/trulens_eval/tracking/logging/where_to_log). See also
    [Database Migration](migration).

  - Offload feedback evaluation using [deferred
    evaluation](/trulens_eval/guides/use_cases_production#deferred-evaluation).
    The evaluator can be deployed to additional computational resources as the
    scale of evals increases. See also
    [start_evaluator][trulens_eval.tru.Tru.start_evaluator].

### 🚧 Customization
  
  - Handle custom apps and custom components via the instrumentation
    decorators and methods. See [Instrumenting LLM
    Applications](/trulens_eval/tracking/instrumentation#instrumenting-llm-applications)
    for what this requires of the user.

  - Evaluate non-wrapped apps or existing logs using virtual apps using
    [TruVirtual][trulens_eval.tru_virtual.TruVirtual]. See also [Running on
    existing
    data](/trulens_eval/evaluation/running_feedback_functions/existing_data).

  - Define [custom providers and feedback
    functions](/trulens_eval/evaluation/feedback_implementations/custom_feedback_functions).
