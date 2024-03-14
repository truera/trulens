LLM applications come in all shapes and sizes and with a variety of different control
flows. As a result it’s a challenge to consistently evaluate parts of an LLM
application trace.

Therefore, we’ve adapted the use of [lenses](https://en.wikipedia.org/wiki/Bidirectional_transformation)
to refer to parts of an LLM stack trace and use those when defining evaluations.
For example, the following lens refers to the input to the retrieve step of the
app called query.

```python
Select.RecordCalls.retrieve.args.query
```

Such lenses can then be used to define evaluations as so:

```python
# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets)
    .aggregate(np.mean)
)
```

In most cases, the Select object produces only a single item but can also
address multiple items.

For example: `Select.RecordCalls.retrieve.args.query` refers to only one item.

However, `Select.RecordCalls.retrieve.rets` refers to multiple items. In this case,
the documents returned by the `retrieve` method. These items can be evaluated separately,
as shown above, or can be collected into an array for evaluation with `.collect()`.
This is most commonly used for groundedness evaluations.

Example:

```python
grounded = Groundedness(groundedness_provider=provider)

f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)
```

Selectors can also access multiple calls to the same component. In agentic applications,
this is an increasingly common practice. For example, an agent could complete multiple
calls to a `retrieve` method to complete the task required.

For example, the following method returns only the returned context documents from
the first invocation of `retrieve`.

```python
context = Select.RecordCalls.retrieve.rets.rets[:]
# Same as context = context_method[0].rets[:]
```

Alternatively, adding `[:]` after the method name `retrieve` returns context documents
from all invocations of `retrieve`.

```python
context_all_calls = Select.RecordCalls.retrieve[:].rets.rets[:]
```

### Understanding the structure of your app

Because LLM apps have a wide variation in their structure, the feedback selector construction
can also vary widely. To construct the feedback selector, you must first understand the structure
of your application.

In python, you can access the JSON structure with `with_record` methods and then calling
`layout_calls_as_app`.

For example:

```python
response = my_llm_app(query)

from trulens_eval import TruChain
tru_recorder = TruChain(
    my_llm_app,
    app_id='Chain1_ChatApplication')

response, tru_record = tru_recorder.with_record(my_llm_app, query)
json_like = tru_record.layout_calls_as_app()
```

If a selector looks like the below

```python
Select.Record.app.combine_documents_chain._call
```

It can be accessed via the JSON-like via

```python
json_like['app']['combine_documents_chain']['_call']
```

The application structure can also be viewed in the TruLens user inerface.
You can view this structure on the `Evaluations` page by scrolling down to the
`Timeline`.

The top level record also contains these helper accessors

- `RecordInput = Record.main_input` -- points to the main input part of a
  Record. This is the first argument to the root method of an app (for
  langchain Chains this is the `__call__` method).

- `RecordOutput = Record.main_output` -- points to the main output part of a
  Record. This is the output of the root method of an app (i.e. `__call__`
  for langchain Chains).

- `RecordCalls = Record.app` -- points to the root of the app-structured
  mirror of calls in a record. See **App-organized Calls** Section above.

## Multiple Inputs Per Argument

As in the `f_qs_relevance` example, a selector for a _single_ argument may point
to more than one aspect of a record/app. These are specified using the slice or
lists in key/index poisitions. In that case, the feedback function is evaluated
multiple times, its outputs collected, and finally aggregated into a main
feedback result.

The collection of values for each argument of feedback implementation is
collected and every combination of argument-to-value mapping is evaluated with a
feedback definition. This may produce a large number of evaluations if more than
one argument names multiple values. In the dashboard, all individual invocations
of a feedback implementation are shown alongside the final aggregate result.

## App/Record Organization (What can be selected)

The top level JSON attributes are defined by the class structures.

For a Record:

```python
class Record(SerialModel):
    record_id: RecordID
    app_id: AppID

    cost: Optional[Cost] = None
    perf: Optional[Perf] = None

    ts: datetime = pydantic.Field(default_factory=lambda: datetime.now())

    tags: str = ""

    main_input: Optional[JSON] = None
    main_output: Optional[JSON] = None  # if no error
    main_error: Optional[JSON] = None  # if error

    # The collection of calls recorded. Note that these can be converted into a
    # json structure with the same paths as the app that generated this record
    # via `layout_calls_as_app`.
    calls: Sequence[RecordAppCall] = []
```

For an App:

```python
class AppDefinition(WithClassInfo, SerialModel, ABC):
    ...

    app_id: AppID

    feedback_definitions: Sequence[FeedbackDefinition] = []

    feedback_mode: FeedbackMode = FeedbackMode.WITH_APP_THREAD

    root_class: Class

    root_callable: ClassVar[FunctionOrMethod]

    app: JSON
```

For your app, you can inspect the JSON-like structure by using the `dict`
method:

```python
tru = ... # your app, extending App
print(tru.dict())
```

### Calls made by App Components

When evaluating a feedback function, Records are augmented with
app/component calls. For example, if the instrumented app
contains a component `combine_docs_chain` then `app.combine_docs_chain` will
contain calls to methods of this component. `app.combine_docs_chain._call` will
contain a `RecordAppCall` (see schema.py) with information about the inputs/outputs/metadata
regarding the `_call` call to that component. Selecting this information is the
reason behind the `Select.RecordCalls` alias.

You can inspect the components making up your app via the `App` method
`print_instrumented`.
