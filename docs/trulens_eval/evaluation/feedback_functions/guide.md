# Specification Guide

## Argument specification

  Several utility methods starting with `.on` provide shorthands:

  - `on_input(arg) == on_prompt(arg: Optional[str])` -- both specify that the next
    unspecified argument or `arg` should be the main app input.

  - `on_output(arg) == on_response(arg: Optional[str])` -- specify that the next
    argument or `arg` should be the main app output.

  - `on_input_output() == on_input().on_output()` -- specifies that the first
    two arguments of implementation should be the main app input and main app
    output, respectively.

  - `on_default()` -- depending on signature of implementation uses either
    `on_output()` if it has a single argument, or `on_input_output` if it has
    two arguments.

  Some wrappers include additional shorthands:

### llama_index-specific selectors

- `TruLlama.select_source_nodes()` -- outputs the selector of the source
    documents part of the engine output.

## Fine-grained Selection and Aggregation

For more advanced control on the feedback function operation, we allow data
selection and aggregation. Consider this feedback example:

```python
f_qs_relevance = Feedback(openai.qs_relevance)
    .on_input()
    .on(Select.Record.app.combine_docs_chain._call.args.inputs.input_documents[:].page_content)
    .aggregate(numpy.min)

# Implementation signature:
# def qs_relevance(self, question: str, statement: str) -> float:
```

- **Argument Selection specification** -- Where we previously set,
  `on_input_output` , the `on(Select...)` line enables specification of where
  the statement argument to the implementation comes from. The form of the
  specification will be discussed in further details in the Specifying Arguments
  section.

- **Aggregation specification** -- The last line `aggregate(numpy.min)` specifies
  how feedback outputs are to be aggregated. This only applies to cases where
  the argument specification names more than one value for an input. The second
  specification, for `statement` was of this type. The input to `aggregate` must
  be a method which can be imported globally. This requirement is further
  elaborated in the next section. This function is called on the `float` results
  of feedback function evaluations to produce a single float. The default is
  `numpy.mean`.

The result of these lines is that `f_qs_relevance` can be now be run on
app/records and will automatically select the specified components of those
apps/records:

```python
record: Record = ...
app: App = ...

feedback_result: FeedbackResult = f_qs_relevance.run(app=app, record=record)
```

The object can also be provided to an app wrapper for automatic evaluation:

```python
app: App = tru.Chain(...., feedbacks=[f_qs_relevance])
```

## Specifying Implementation Function and Aggregate

The function or method provided to the `Feedback` constructor is the
implementation of the feedback function which does the actual work of producing
a float indicating some quantity of interest. 

**Note regarding FeedbackMode.DEFERRED** -- Any function or method (not static
or class methods presently supported) can be provided here but there are
additional requirements if your app uses the "deferred" feedback evaluation mode
(when `feedback_mode=FeedbackMode.DEFERRED` are specified to app constructor).
In those cases the callables must be functions or methods that are importable
(see the next section for details). The function/method performing the
aggregation has the same requirements.

### Import requirement (DEFERRED feedback mode only)

If using deferred evaluation, the feedback function implementations and
aggregation implementations must be functions or methods from a Provider
subclass that is importable. That is, the callables must be accessible were you
to evaluate this code:

```python
from somepackage.[...] import someproviderclass
from somepackage.[...] import somefunction

# [...] means optionally further package specifications

provider = someproviderclass(...) # constructor arguments can be included
feedback_implementation1 = provider.somemethod
feedback_implementation2 = somefunction
```

For provided feedback functions, `somepackage` is `trulens_eval.feedback` and
`someproviderclass` is `OpenAI` or one of the other `Provider` subclasses.
Custom feedback functions likewise need to be importable functions or methods of
a provider subclass that can be imported. Critically, functions or classes
defined locally in a notebook will not be importable this way.

## Specifying Arguments

The mapping between app/records to feedback implementation arguments is
specified by the `on...` methods of the `Feedback` objects. The general form is:

```python
feedback: Feedback = feedback.on(argname1=selector1, argname2=selector2, ...)
```

That is, `Feedback.on(...)` returns a new `Feedback` object with additional
argument mappings, the source of `argname1` is `selector1` and so on for further
argument names. The types of `selector1` is `JSONPath` which we elaborate on in
the "Selector Details".

If argument names are ommitted, they are taken from the feedback function
implementation signature in order. That is, 

```python
Feedback(...).on(argname1=selector1, argname2=selector2)
```

and

```python
Feedback(...).on(selector1, selector2)
```

are equivalent assuming the feedback implementation has two arguments,
`argname1` and `argname2`, in that order.

### Running Feedback

Feedback implementations are simple callables that can be run on any arguments
matching their signatures. However, once wrapped with `Feedback`, they are meant
to be run on outputs of app evaluation (the "Records"). Specifically,
`Feedback.run` has this definition:

```python
def run(self, 
    app: Union[AppDefinition, JSON], 
    record: Record
) -> FeedbackResult:
```

That is, the context of a Feedback evaluation is an app (either as
`AppDefinition` or a JSON-like object) and a `Record` of the execution of the
aforementioned app. Both objects are indexable using "Selectors". By indexable
here we mean that their internal components can be specified by a Selector and
subsequently that internal component can be extracted using that selector.
Selectors for Feedback start by specifying whether they are indexing into an App
or a Record via the `__app__` and `__record__` special
attributes (see **Selectors** section below).

### Selector Details

Apps and Records will be converted to JSON-like structures representing their callstack.

Selectors are of type `JSONPath` defined in `utils/serial.py` help specify paths into JSON-like
structures (enumerating `Record` or `App` contents). 

In most cases, the Select object produces only a single item but can also
address multiple items.

You can access the JSON structure with `with_record` methods and then calling `layout_calls_as_app`.

for example

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
