# trulens-feedback

## Feedback Functions

The `Feedback` class contains the starting point for feedback function
specification and evaluation. A typical use-case looks like this:

```python
from trulens.core import Feedback, Select, Feedback

hugs = feedback.Huggingface()

f_lang_match = Feedback(hugs.language_match)
    .on_input_output()
```

The components of this specifications are:

- **Provider classes** -- `feedback.OpenAI` contains feedback function
  implementations like `context_relevance`. Other classes subtyping
  `feedback.Provider` include `HuggingFace` and `Cohere`.

- **Feedback implementations** -- `provider.context_relevance` is a feedback function
  implementation. Feedback implementations are simple callables that can be run
  on any arguments matching their signatures. In the example, the implementation
  has the following signature:

  ```python
  def language_match(self, text1: str, text2: str) -> float:
  ```

  That is, `language_match` is a plain Python method that accepts two pieces
  of text, both strings, and produces a float (assumed to be between 0.0 and
  1.0).

- **Feedback constructor** -- The line `Feedback(provider.language_match)`
  constructs a Feedback object with a feedback implementation.

- **Argument specification** -- The next line, `on_input_output`, specifies how
  the `language_match` arguments are to be determined from an app record or app
  definition. The general form of this specification is done using `on` but
  several shorthands are provided. `on_input_output` states that the first two
  argument to `language_match` (`text1` and `text2`) are to be the main app
  input and the main output, respectively.

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

### LlamaIndex-specific selectors

  - `TruLlama.select_source_nodes()` -- outputs the selector for the source
    documents part of the engine output.
  - `TruLlama.select_context()` -- outputs the selector for the text of
    the source documents part of the engine output.

### LangChain-specific selectors

  - `TruChain.select_context()` -- outputs the selector for retrieved context
    from the app's internal `get_relevant_documents` method.

### NeMo-specific selectors

  - `NeMo.select_context()` -- outputs the selector for the retrieved context
    from the app's internal `search_relevant_chunks` method.


## Fine-grained Selection and Aggregation

For more advanced control on the feedback function operation, we allow data
selection and aggregation. Consider this feedback example:

```python
f_context_relevance = Feedback(openai.context_relevance)
    .on_input()
    .on(Select.Record.app.combine_docs_chain._call.args.inputs.input_documents[:].page_content)
    .aggregate(numpy.mean)

# Implementation signature:
# def context_relevance(self, question: str, statement: str) -> float:
```

- **Argument Selection specification** -- Where we previously set,
  `on_input_output` , the `on(Select...)` line enables specification of where
  the statement argument to the implementation comes from. The form of the
  specification will be discussed in further details in the Specifying Arguments
  section.

- **Aggregation specification** -- The last line `aggregate(numpy.mean)` specifies
  how feedback outputs are to be aggregated. This only applies to cases where
  the argument specification names more than one value for an input. The second
  specification, for `context` was of this type. The input to `aggregate` must
  be a method which can be imported globally. This requirement is further
  elaborated in the next section. This function is called on the `float` results
  of feedback function evaluations to produce a single float. The default is
  `numpy.mean`.

The result of these lines is that `f_context_relevance` can be now be run on
app/records and will automatically select the specified components of those
apps/records:

```python
record: Record = ...
app: App = ...

feedback_result: FeedbackResult = f_context_relevance.run(app=app, record=record)
```

The object can also be provided to an app wrapper for automatic evaluation:

```python
app: App = TruChain(...., feedbacks=[f_context_relevance])
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

For provided feedback functions, `somepackage` is `trulens.feedback` and
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

If argument names are omitted, they are taken from the feedback function
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

Selectors are of type `JSONPath` defined in `util.py` but are also aliased in
`schema.py` as `Select.Query`. Objects of this type specify paths into JSON-like
structures (enumerating `Record` or `App` contents).

By JSON-like structures we mean Python objects that can be converted into JSON
or are base types. This includes:

- base types: strings, integers, dates, etc.

- sequences

- dictionaries with string keys

Additionally, JSONPath also index into general Python objects like
`AppDefinition` or `Record` though each of these can be converted to JSON-like.

When used to index json-like objects, JSONPath are used as generators: the path
can be used to iterate over items from within the object:

```python
class JSONPath...
    ...
    def __call__(self, obj: Any) -> Iterable[Any]:
    ...
```

In most cases, the generator produces only a single item but paths can also
address multiple items (as opposed to a single item containing multiple).

The syntax of this specification mirrors the syntax one would use with
instantiations of JSON-like objects. For every `obj` generated by `query: JSONPath`:

- `query[somekey]` generates the `somekey` element of `obj` assuming it is a
  dictionary with key `somekey`.

- `query[someindex]` generates the index `someindex` of `obj` assuming it is
  a sequence.

- `query[slice]` generates the **multiple** elements of `obj` assuming it is a
  sequence. Slices include `:` or in general `startindex:endindex:step`.

- `query[somekey1, somekey2, ...]` generates **multiple** elements of `obj`
  assuming `obj` is a dictionary and `somekey1`... are its keys.

- `query[someindex1, someindex2, ...]` generates **multiple** elements
  indexed by `someindex1`... from a sequence `obj`.

- `query.someattr` depends on type of `obj`. If `obj` is a dictionary, then
  `query.someattr` is an alias for `query[someattr]`. Otherwise if
  `someattr` is an attribute of a Python object `obj`, then `query.someattr`
  generates the named attribute.

For feedback argument specification, the selectors should start with either
`__record__` or `__app__` indicating which of the two JSON-like structures to
select from (Records or Apps). `Select.Record` and `Select.App` are defined as
`Query().__record__` and `Query().__app__` and thus can stand in for the start of a
selector specification that wishes to select from a Record or App, respectively.
The full set of Query aliases are as follows:

- `Record = Query().__record__` -- points to the Record.

- App = Query().**app** -- points to the App.

- `RecordInput = Record.main_input` -- points to the main input part of a
  Record. This is the first argument to the root method of an app (for
  LangChain Chains this is the `__call__` method).

- `RecordOutput = Record.main_output` -- points to the main output part of a
  Record. This is the output of the root method of an app (i.e. `__call__`
  for LangChain Chains).

- `RecordCalls = Record.app` -- points to the root of the app-structured
  mirror of calls in a record. See **App-organized Calls** Section above.

## Multiple Inputs Per Argument

As in the `f_context_relevance` example, a selector for a _single_ argument may point
to more than one aspect of a record/app. These are specified using the slice or
lists in key/index positions. In that case, the feedback function is evaluated
multiple times, its outputs collected, and finally aggregated into a main
feedback result.

The collection of values for each argument of feedback implementation is
collected and every combination of argument-to-value mapping is evaluated with a
feedback definition. This may produce a large number of evaluations if more than
one argument names multiple values. In the dashboard, all individual invocations
of a feedback implementation are shown alongside the final aggregate result.

## App/Record Organization (What can be selected)

Apps are serialized into JSON-like structures which are indexed via selectors.
The exact makeup of this structure is app-dependent though always start with
`app`, that is, the trulens wrappers (subtypes of `App`) contain the wrapped app
in the attribute `app`:

```python
# app.py:
class App(AppDefinition, SerialModel):
    ...
    # The wrapped app.
    app: Any = Field(exclude=True)
    ...
```

For your app, you can inspect the JSON-like structure by using the `dict`
method:

```python
app = ... # your app, extending App
print(app.dict())
```

The other non-excluded fields accessible outside of the wrapped app are listed
in the `AppDefinition` class in `schema.py`:

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

Note that `app` is in both classes. This distinction between `App` and
`AppDefinition` here is that one corresponds to potentially non-serializable
Python objects (`App`) and their serializable versions (`AppDefinition`).
Feedbacks should expect to be run with `AppDefinition`. Fields of `App` that are
not part of `AppDefinition` may not be available.

You can inspect the data available for feedback definitions in the dashboard by
clicking on the "See full app json" button on the bottom of the page after
selecting a record from a table.

The other piece of context to Feedback evaluation are records. These contain the
inputs/outputs and other information collected during the execution of an app:

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

A listing of a record can be seen in the dashboard by clicking the "see full
record json" button on the bottom of the page after selecting a record from the
table.

### Calls made by App Components

When evaluating a feedback function, Records are augmented with
app/component calls in app layout in the attribute `app`. By this we mean that
in addition to the fields listed in the class definition above, the `app` field
will contain the same information as `calls` but organized in a manner mirroring
the organization of the app structure. For example, if the instrumented app
contains a component `combine_docs_chain` then `app.combine_docs_chain` will
contain calls to methods of this component. In the example at the top of this
docstring, `_call` was an example of such a method. Thus
`app.combine_docs_chain._call` further contains a `RecordAppCall` (see
schema.py) structure with information about the inputs/outputs/metadata
regarding the `_call` call to that component. Selecting this information is the
reason behind the `Select.RecordCalls` alias (see next section).

You can inspect the components making up your app via the `App` method
`print_instrumented`.
