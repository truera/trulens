"""
# Feedback Functions

The `Feedback` class contains the starting point for feedback function
specification and evaluation. A typical use-case looks like this:

```python 
from trulens_eval import feedback, Select, Feedback

hugs = feedback.Huggingface()

f_lang_match = Feedback(hugs.language_match)
    .on_input_output()
```

The components of this specifications are:

- **Provider classes** -- `feedback.OpenAI` contains feedback function
  implementations like `qs_relevance`. Other classes subtyping
  `feedback.Provider` include `Huggingface` and `Cohere`.

- **Feedback implementations** -- `openai.qs_relevance` is a feedback function
  implementation. Feedback implementations are simple callables that can be run
  on any arguments matching their signatures. In the example, the implementation
  has the following signature: 

    ```python
    def language_match(self, text1: str, text2: str) -> float:
    ```

  That is, `language_match` is a plain python method that accepts two pieces
  of text, both strings, and produces a float (assumed to be between 0.0 and
  1.0).

- **Feedback constructor** -- The line `Feedback(openai.language_match)`
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

- **Argument Selection specification ** -- Where we previously set,
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

Selectors are of type `JSONPath` defined in `util.py` but are also aliased in
`schema.py` as `Select.Query`. Objects of this type specify paths into JSON-like
structures (enumerating `Record` or `App` contents). 

By JSON-like structures we mean python objects that can be converted into JSON
or are base types. This includes:

- base types: strings, integers, dates, etc.

- sequences

- dictionaries with string keys

Additionally, JSONPath also index into general python objects like
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

- `query[slice]` generates the __multiple__ elements of `obj` assuming it is a
    sequence. Slices include `:` or in general `startindex:endindex:step`.

- `query[somekey1, somekey2, ...]` generates __multiple__ elements of `obj`
    assuming `obj` is a dictionary and `somekey1`... are its keys.

- `query[someindex1, someindex2, ...]` generates __multiple__ elements
    indexed by `someindex1`... from a sequence `obj`.

- `query.someattr` depends on type of `obj`. If `obj` is a dictionary, then
    `query.someattr` is an alias for `query[someattr]`. Otherwise if
    `someattr` is an attribute of a python object `obj`, then `query.someattr`
    generates the named attribute.

For feedback argument specification, the selectors should start with either
`__record__` or `__app__` indicating which of the two JSON-like structures to
select from (Records or Apps). `Select.Record` and `Select.App` are defined as
`Query().__record__` and `Query().__app__` and thus can stand in for the start of a
selector specification that wishes to select from a Record or App, respectively.
The full set of Query aliases are as follows:

- `Record = Query().__record__` -- points to the Record.

- App = Query().__app__ -- points to the App.

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
tru = ... # your app, extending App
print(tru.dict())
```

The other non-excluded fields accessible outside of the wrapped app are listed
in the `AppDefinition` class in `schema.py`:

```python
class AppDefinition(SerialModel, WithClassInfo, ABC):
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
python objects (`App`) and their serializable versions (`AppDefinition`).
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
"""

from datetime import datetime
from inspect import Signature
from inspect import signature
import itertools
import logging
from multiprocessing.pool import AsyncResult
import re
import traceback
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
)

import numpy as np
import openai
import pydantic
from tqdm import tqdm

from trulens_eval import feedback_prompts
from trulens_eval.keys import *
from trulens_eval.provider_apis import Endpoint
from trulens_eval.provider_apis import HuggingfaceEndpoint
from trulens_eval.provider_apis import OpenAIEndpoint
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackCall
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import FeedbackResultID
from trulens_eval.schema import FeedbackResultStatus
from trulens_eval.schema import Record
from trulens_eval.schema import Select
from trulens_eval.util import FunctionOrMethod
from trulens_eval.util import JSON
from trulens_eval.util import jsonify
from trulens_eval.util import SerialModel
from trulens_eval.util import TP
from trulens_eval.util import WithClassInfo
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_CLOCK
from trulens_eval.utils.text import UNICODE_YIELD

PROVIDER_CLASS_NAMES = ['OpenAI', 'Huggingface', 'Cohere']

logger = logging.getLogger(__name__)


def check_provider(cls_or_name: Union[Type, str]) -> None:
    if isinstance(cls_or_name, str):
        cls_name = cls_or_name
    else:
        cls_name = cls_or_name.__name__

    assert cls_name in PROVIDER_CLASS_NAMES, f"Unsupported provider class {cls_name}"


# Signature of feedback implementations. Take in any number of arguments
# and return either a single float or a float and a dictionary (of metadata).
ImpCallable = Callable[..., Union[float, Tuple[float, Dict[str, Any]]]]

# Signature of aggregation functions.
AggCallable = Callable[[Iterable[float]], float]


class Feedback(FeedbackDefinition):
    # Implementation, not serializable, note that FeedbackDefinition contains
    # `implementation` meant to serialize the below.
    imp: Optional[ImpCallable] = pydantic.Field(exclude=True)

    # Aggregator method for feedback functions that produce more than one
    # result.
    agg: Optional[AggCallable] = pydantic.Field(exclude=True)

    def __init__(
        self,
        imp: Optional[Callable] = None,
        agg: Optional[Callable] = None,
        **kwargs
    ):
        """
        A Feedback function container.

        Parameters:
        
        - imp: Optional[Callable] -- implementation of the feedback function.

        - agg: Optional[Callable] -- aggregation function for producing a single
          float for feedback implementations that are run more than once.
        """

        agg = agg or np.mean

        # imp is the python function/method while implementation is a serialized
        # json structure. Create the one that is missing based on the one that
        # is provided:

        if imp is not None:
            # These are for serialization to/from json and for db storage.
            if 'implementation' not in kwargs:
                try:
                    kwargs['implementation'] = FunctionOrMethod.of_callable(
                        imp, loadable=True
                    )
                except ImportError as e:
                    logger.warning(
                        f"Feedback implementation {imp} cannot be serialized: {e}. "
                        f"This may be ok unless you are using the deferred feedback mode."
                    )

                    kwargs['implementation'] = FunctionOrMethod.of_callable(
                        imp, loadable=False
                    )

        else:
            if "implementation" in kwargs:
                imp: ImpCallable = FunctionOrMethod.pick(
                    **(kwargs['implementation'])
                ).load() if kwargs['implementation'] is not None else None

        # Similarly with agg and aggregator.
        if agg is not None:
            if 'aggregator' not in kwargs:
                try:
                    # These are for serialization to/from json and for db storage.
                    kwargs['aggregator'] = FunctionOrMethod.of_callable(
                        agg, loadable=True
                    )
                except:
                    # User defined functions in script do not have a module so cannot be serialized
                    pass
        else:
            if 'aggregator' in kwargs:
                agg: AggCallable = FunctionOrMethod.pick(
                    **(kwargs['aggregator'])
                ).load()

        super().__init__(**kwargs)

        self.imp = imp
        self.agg = agg

        # Verify that `imp` expects the arguments specified in `selectors`:
        if self.imp is not None:
            sig: Signature = signature(self.imp)
            for argname in self.selectors.keys():
                assert argname in sig.parameters, (
                    f"{argname} is not an argument to {self.imp.__name__}. "
                    f"Its arguments are {list(sig.parameters.keys())}."
                )

    def on_input_output(self):
        """
        Specifies that the feedback implementation arguments are to be the main
        app input and output in that order.

        Returns a new Feedback object with the specification.
        """
        return self.on_input().on_output()

    def on_default(self):
        """
        Specifies that one argument feedbacks should be evaluated on the main
        app output and two argument feedbacks should be evaluates on main input
        and main output in that order.

        Returns a new Feedback object with this specification.
        """

        ret = Feedback().parse_obj(self)
        ret._default_selectors()
        return ret

    def _print_guessed_selector(self, par_name, par_path):
        if par_path == Select.RecordCalls:
            alias_info = f" or `Select.RecordCalls`"
        elif par_path == Select.RecordInput:
            alias_info = f" or `Select.RecordInput`"
        elif par_path == Select.RecordOutput:
            alias_info = f" or `Select.RecordOutput`"
        else:
            alias_info = ""

        print(
            f"{UNICODE_CHECK} In {self.name}, "
            f"input {par_name} will be set to {par_path}{alias_info} ."
        )

    def _default_selectors(self):
        """
        Fill in default selectors for any remaining feedback function arguments.
        """

        assert self.imp is not None, "Feedback function implementation is required to determine default argument names."

        sig: Signature = signature(self.imp)
        par_names = list(
            k for k in sig.parameters.keys() if k not in self.selectors
        )

        if len(par_names) == 1:
            # A single argument remaining. Assume it is record output.
            selectors = {par_names[0]: Select.RecordOutput}
            self._print_guessed_selector(par_names[0], Select.RecordOutput)

            # TODO: replace with on_output ?

        elif len(par_names) == 2:
            # Two arguments remaining. Assume they are record input and output
            # respectively.
            selectors = {
                par_names[0]: Select.RecordInput,
                par_names[1]: Select.RecordOutput
            }
            self._print_guessed_selector(par_names[0], Select.RecordInput)
            self._print_guessed_selector(par_names[1], Select.RecordOutput)

            # TODO: replace on_input_output ?
        else:
            # Otherwise give up.

            raise RuntimeError(
                f"Cannot determine default paths for feedback function arguments. "
                f"The feedback function has signature {sig}."
            )

        self.selectors = selectors

    @staticmethod
    def evaluate_deferred(tru: 'Tru') -> int:
        """
        Evaluates feedback functions that were specified to be deferred. Returns
        an integer indicating how many evaluates were run.
        """

        db = tru.db

        def prepare_feedback(row):
            record_json = row.record_json
            record = Record(**record_json)

            app_json = row.app_json

            feedback = Feedback(**row.feedback_json)
            feedback.run_and_log(
                record=record,
                app=app_json,
                tru=tru,
                feedback_result_id=row.feedback_result_id
            )

        feedbacks = db.get_feedback()

        started_count = 0

        for i, row in feedbacks.iterrows():
            feedback_ident = f"{row.fname} for app {row.app_json['app_id']}, record {row.record_id}"

            if row.status == FeedbackResultStatus.NONE:

                print(
                    f"{UNICODE_YIELD} Feedback task starting: {feedback_ident}"
                )

                TP().runlater(prepare_feedback, row)
                started_count += 1

            elif row.status in [FeedbackResultStatus.RUNNING]:
                now = datetime.now().timestamp()
                if now - row.last_ts > 30:
                    print(
                        f"{UNICODE_YIELD} Feedback task last made progress over 30 seconds ago. "
                        f"Retrying: {feedback_ident}"
                    )
                    TP().runlater(prepare_feedback, row)
                    started_count += 1

                else:
                    print(
                        f"{UNICODE_CLOCK} Feedback task last made progress less than 30 seconds ago. "
                        f"Giving it more time: {feedback_ident}"
                    )

            elif row.status in [FeedbackResultStatus.FAILED]:
                now = datetime.now().timestamp()
                if now - row.last_ts > 60 * 5:
                    print(
                        f"{UNICODE_YIELD} Feedback task last made progress over 5 minutes ago. "
                        f"Retrying: {feedback_ident}"
                    )
                    TP().runlater(prepare_feedback, row)
                    started_count += 1

                else:
                    print(
                        f"{UNICODE_CLOCK} Feedback task last made progress less than 5 minutes ago. "
                        f"Not touching it for now: {feedback_ident}"
                    )

            elif row.status == FeedbackResultStatus.DONE:
                pass

        return started_count

    def __call__(self, *args, **kwargs) -> Any:
        assert self.imp is not None, "Feedback definition needs an implementation to call."
        return self.imp(*args, **kwargs)

    def aggregate(self, func: Callable) -> 'Feedback':
        """
        Specify the aggregation function in case the selectors for this feedback
        generate more than one value for implementation argument(s).

        Returns a new Feedback object with the given aggregation function.
        """

        return Feedback(imp=self.imp, selectors=self.selectors, agg=func)

    @staticmethod
    def of_feedback_definition(f: FeedbackDefinition):
        implementation = f.implementation
        aggregator = f.aggregator

        imp_func = implementation.load()
        agg_func = aggregator.load()

        return Feedback(imp=imp_func, agg=agg_func, **f.dict())

    def _next_unselected_arg_name(self):
        if self.imp is not None:
            sig = signature(self.imp)
            par_names = list(
                k for k in sig.parameters.keys() if k not in self.selectors
            )
            if "self" in par_names:
                logger.warning(
                    f"Feedback function `{self.imp.__name__}` has `self` as argument. "
                    "Perhaps it is static method or its Provider class was not initialized?"
                )
            return par_names[0]
        else:
            raise RuntimeError(
                "Cannot determine name of feedback function parameter without its definition."
            )

    def on_prompt(self, arg: Optional[str] = None):
        """
        Create a variant of `self` that will take in the main app input or
        "prompt" as input, sending it as an argument `arg` to implementation.
        """

        new_selectors = self.selectors.copy()

        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, Select.RecordInput)

        new_selectors[arg] = Select.RecordInput

        return Feedback(imp=self.imp, selectors=new_selectors, agg=self.agg)

    on_input = on_prompt

    def on_response(self, arg: Optional[str] = None):
        """
        Create a variant of `self` that will take in the main app output or
        "response" as input, sending it as an argument `arg` to implementation.
        """

        new_selectors = self.selectors.copy()

        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, Select.RecordOutput)

        new_selectors[arg] = Select.RecordOutput

        return Feedback(imp=self.imp, selectors=new_selectors, agg=self.agg)

    on_output = on_response

    def on(self, *args, **kwargs):
        """
        Create a variant of `self` with the same implementation but the given
        selectors. Those provided positionally get their implementation argument
        name guessed and those provided as kwargs get their name from the kwargs
        key.
        """

        new_selectors = self.selectors.copy()
        new_selectors.update(kwargs)

        for path in args:
            argname = self._next_unselected_arg_name()
            new_selectors[argname] = path
            self._print_guessed_selector(argname, path)

        return Feedback(imp=self.imp, selectors=new_selectors, agg=self.agg)

    def run(
        self, app: Union[AppDefinition, JSON], record: Record
    ) -> FeedbackResult:
        """
        Run the feedback function on the given `record`. The `app` that
        produced the record is also required to determine input/output argument
        names.

        Might not have a AppDefinitionhere but only the serialized app_json .
        """

        if isinstance(app, AppDefinition):
            app_json = jsonify(app)
        else:
            app_json = app

        result_vals = []

        feedback_calls = []

        feedback_result = FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            record_id=record.record_id,
            name=self.name
        )

        try:
            # Total cost, will accumulate.
            cost = Cost()

            for ins in self.extract_selection(app=app_json, record=record):

                result_and_meta, part_cost = Endpoint.track_all_costs_tally(
                    lambda: self.imp(**ins)
                )
                cost += part_cost

                if isinstance(result_and_meta, Tuple):
                    # If output is a tuple of two, we assume it is the float and the metadata.
                    assert len(
                        result_and_meta
                    ) == 2, "Feedback functions must return either a single float or a float and a dictionary."
                    result_val, meta = result_and_meta

                    assert isinstance(
                        meta, dict
                    ), f"Feedback metadata output must be a dictionary but was {type(call_meta)}."
                else:
                    # Otherwise it is just the float. We create empty metadata dict.
                    result_val = result_and_meta
                    meta = dict()

                if isinstance(result_val, dict):
                    for val in result_val.values():
                        assert isinstance(
                            val, float
                        ), f"Feedback function output with multivalue must be a dict with float values but encountered {type(val)}."
                    # TODO: Better handling of multi-output
                    result_val = list(result_val.values())
                    feedback_call = FeedbackCall(
                        args=ins, ret=np.mean(result_val), meta=meta
                    )

                else:
                    assert isinstance(
                        result_val, float
                    ), f"Feedback function output must be a float or dict but was {type(result_val)}."
                    feedback_call = FeedbackCall(
                        args=ins, ret=result_val, meta=meta
                    )

                result_vals.append(result_val)
                feedback_calls.append(feedback_call)

            result_vals = np.array(result_vals)
            if len(result_vals) == 0:
                logger.warning(
                    f"Feedback function {self.name} with aggregation {self.agg} had no inputs."
                )
                result = np.nan
            else:
                result = self.agg(result_vals)

            feedback_result.update(
                result=result,
                status=FeedbackResultStatus.DONE,
                cost=cost,
                calls=feedback_calls
            )

            return feedback_result

        except:
            exc_tb = traceback.format_exc()
            logger.warning(f"Feedback Function Exception Caught: {exc_tb}")
            feedback_result.update(
                error=exc_tb, status=FeedbackResultStatus.FAILED
            )
            return feedback_result

    def run_and_log(
        self,
        record: Record,
        tru: 'Tru',
        app: Union[AppDefinition, JSON] = None,
        feedback_result_id: Optional[FeedbackResultID] = None
    ) -> FeedbackResult:
        record_id = record.record_id
        app_id = record.app_id

        db = tru.db

        # Placeholder result to indicate a run.
        feedback_result = FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            feedback_result_id=feedback_result_id,
            record_id=record_id,
            name=self.name
        )

        if feedback_result_id is None:
            feedback_result_id = feedback_result.feedback_result_id

        try:
            db.insert_feedback(
                feedback_result.update(
                    status=FeedbackResultStatus.RUNNING  # in progress
                )
            )

            feedback_result = self.run(
                app=app, record=record
            ).update(feedback_result_id=feedback_result_id)

        except Exception as e:
            exc_tb = traceback.format_exc()
            db.insert_feedback(
                feedback_result.update(
                    error=exc_tb, status=FeedbackResultStatus.FAILED
                )
            )
            return

        # Otherwise update based on what Feedback.run produced (could be success or failure).
        db.insert_feedback(feedback_result)

        return feedback_result

    @property
    def name(self):
        """
        Name of the feedback function. Presently derived from the name of the
        function implementing it.
        """

        if self.imp is None:
            raise RuntimeError("This feedback function has no implementation.")

        return self.imp.__name__

    def extract_selection(
        self, app: Union[AppDefinition, JSON], record: Record
    ) -> Iterable[Dict[str, Any]]:
        """
        Given the `app` that produced the given `record`, extract from
        `record` the values that will be sent as arguments to the implementation
        as specified by `self.selectors`.
        """

        arg_vals = {}

        for k, v in self.selectors.items():
            if isinstance(v, Select.Query):
                q = v

            else:
                raise RuntimeError(f"Unhandled selection type {type(v)}.")

            if q.path[0] == Select.Record.path[0]:
                o = record.layout_calls_as_app()
            elif q.path[0] == Select.App.path[0]:
                o = app
            else:
                raise ValueError(
                    f"Query {q} does not indicate whether it is about a record or about a app."
                )

            q_within_o = Select.Query(path=q.path[1:])
            arg_vals[k] = list(q_within_o(o))

        keys = arg_vals.keys()
        vals = arg_vals.values()

        assignments = itertools.product(*vals)

        for assignment in assignments:
            yield {k: v for k, v in zip(keys, assignment)}


pat_1_10 = re.compile(r"\s*([1-9][0-9]*)\s*")


def _re_1_10_rating(str_val):
    matches = pat_1_10.fullmatch(str_val)
    if not matches:
        # Try soft match
        matches = re.search('[1-9][0-9]*', str_val)
        if not matches:
            logger.warn(f"1-10 rating regex failed to match on: '{str_val}'")
            return -10  # so this will be reported as -1 after division by 10

    return int(matches.group())


class Provider(SerialModel, WithClassInfo):

    class Config:
        arbitrary_types_allowed = True

    endpoint: Optional[Endpoint]

    def __init__(self, *args, **kwargs):
        # for WithClassInfo:
        kwargs['obj'] = self

        super().__init__(*args, **kwargs)


class OpenAI(Provider):
    model_engine: str
    endpoint: Endpoint

    def __init__(
        self, *args, endpoint=None, model_engine="gpt-3.5-turbo", **kwargs
    ):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        A set of OpenAI Feedback Functions.

        Parameters:

        - model_engine (str, optional): The specific model version. Defaults to
          "gpt-3.5-turbo".

        - All other args/kwargs passed to OpenAIEndpoint constructor.
        """

        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict()
        self_kwargs['model_engine'] = model_engine
        self_kwargs['endpoint'] = OpenAIEndpoint(*args, **kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

        set_openai_key()

    """
    def to_json(self) -> Dict:
        return Provider.to_json(self, model_engine=self.model_engine)
    """

    def _create_chat_completion(self, *args, **kwargs):
        return openai.ChatCompletion.create(*args, **kwargs)

    def _moderation(self, text: str):
        return self.endpoint.run_me(
            lambda: openai.Moderation.create(input=text)
        )

    def moderation_not_hate(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is hate
        speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "hate" and 1 being "not
            hate".
        """
        openai_response = self._moderation(text)
        return 1 - float(
            openai_response["results"][0]["category_scores"]["hate"]
        )

    def moderation_not_hatethreatening(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is
        threatening speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "threatening" and 1 being
            "not threatening".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["hate/threatening"]
        )

    def moderation_not_selfharm(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        self harm.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "self harm" and 1 being "not
            self harm".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["self-harm"]
        )

    def moderation_not_sexual(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is sexual
        speech.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "sexual" and 1 being "not
            sexual".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual"]
        )

    def moderation_not_sexualminors(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        sexual minors.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "sexual minors" and 1 being
            "not sexual minors".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["sexual/minors"]
        )

    def moderation_not_violence(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        violence.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "violence" and 1 being "not
            violence".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence"]
        )

    def moderation_not_violencegraphic(self, text: str) -> float:
        """
        Uses OpenAI's Moderation API. A function that checks if text is about
        graphic violence.

        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "graphic violence" and 1
            being "not graphic violence".
        """
        openai_response = self._moderation(text)

        return 1 - int(
            openai_response["results"][0]["category_scores"]["violence/graphic"]
        )

    def _find_relevant_string(self, full_source, hypothesis):
        return self.endpoint.run_me(
            lambda: self._create_chat_completion(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            str.format(
                                feedback_prompts.SYSTEM_FIND_SUPPORTING,
                                prompt=full_source,
                            )
                    }, {
                        "role":
                            "user",
                        "content":
                            str.format(
                                feedback_prompts.USER_FIND_SUPPORTING,
                                response=hypothesis
                            )
                    }
                ]
            )["choices"][0]["message"]["content"]
        )

    def _summarized_groundedness(self, premise: str, hypothesis: str) -> float:
        """ A groundedness measure best used for summarized premise against simple hypothesis.
        This OpenAI implementation uses information overlap prompts.

        Args:
            premise (str): Summarized source sentences.
            hypothesis (str): Single statement setnece.

        Returns:
            float: Information Overlap
        """
        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    feedback_prompts.LLM_GROUNDEDNESS,
                                    premise=premise,
                                    hypothesis=hypothesis,
                                )
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def _groundedness_doc_in_out(self, premise: str, hypothesis: str) -> str:
        """An LLM prompt using the entire document for premise and entire statement document for hypothesis

        Args:
            premise (str): A source document
            hypothesis (str): A statement to check

        Returns:
            str: An LLM response using a scorecard template
        """
        return self.endpoint.run_me(
            lambda: self._create_chat_completion(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            str.format(
                                feedback_prompts.LLM_GROUNDEDNESS_FULL_SYSTEM,
                            )
                    }, {
                        "role":
                            "user",
                        "content":
                            str.format(
                                feedback_prompts.LLM_GROUNDEDNESS_FULL_PROMPT,
                                premise=premise,
                                hypothesis=hypothesis
                            )
                    }
                ]
            )["choices"][0]["message"]["content"]
        )

    def qs_relevance(self, question: str, statement: str) -> float:
        """
        Uses OpenAI's Chat Completion App. A function that completes a
        template to check the relevance of the statement to the question.

        Parameters:
            question (str): A question being asked. statement (str): A statement
            to the question.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    feedback_prompts.QS_RELEVANCE,
                                    question=question,
                                    statement=statement
                                )
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def relevance(self, prompt: str, response: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the relevance of the response to a prompt.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being
            "relevant".
        """
        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                str.format(
                                    feedback_prompts.PR_RELEVANCE,
                                    prompt=prompt,
                                    response=response
                                )
                        },
                    ]
                )["choices"][0]["message"]["content"]
            )
        ) / 10

    def sentiment(self, text: str) -> float:
        """
        Uses OpenAI's Chat Completion Model. A function that completes a
        template to check the sentiment of some text.

        Parameters:
            text (str): A prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """

        return _re_1_10_rating(
            self.endpoint.run_me(
                lambda: self._create_chat_completion(
                    model=self.model_engine,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": feedback_prompts.SENTIMENT_SYSTEM_PROMPT
                        }, {
                            "role": "user",
                            "content": text
                        }
                    ]
                )["choices"][0]["message"]["content"]
            )
        )

    def model_agreement(self, prompt: str, response: str) -> float:
        """
        Uses OpenAI's Chat GPT Model. A function that gives Chat GPT the same
        prompt and gets a response, encouraging truthfulness. A second template
        is given to Chat GPT with a prompt that the original response is
        correct, and measures whether previous Chat GPT's response is similar.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The agent's
            response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not in agreement" and 1
            being "in agreement".
        """
        logger.warning(
            "model_agreement has been deprecated. Use GroundTruthAgreement(ground_truth) instead."
        )
        oai_chat_response = self.endpoint.run_me(
            lambda: self._create_chat_completion(
                model=self.model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": feedback_prompts.CORRECT_SYSTEM_PROMPT
                    }, {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )["choices"][0]["message"]["content"]
        )
        agreement_txt = self._get_answer_agreement(
            prompt, response, oai_chat_response, self.model_engine
        )
        return _re_1_10_rating(agreement_txt) / 10

    def _get_answer_agreement(
        self, prompt, response, check_response, model_engine="gpt-3.5-turbo"
    ):
        oai_chat_response = self.endpoint.run_me(
            lambda: self._create_chat_completion(
                model=model_engine,
                temperature=0.0,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            feedback_prompts.AGREEMENT_SYSTEM_PROMPT %
                            (prompt, response)
                    }, {
                        "role": "user",
                        "content": check_response
                    }
                ]
            )["choices"][0]["message"]["content"]
        )
        return oai_chat_response


class Groundedness(SerialModel, WithClassInfo):
    summarize_provider: Provider
    groundedness_provider: Provider

    def __init__(self, groundedness_provider: Provider = None):
        """Instantiates the groundedness providers. Currently the groundedness functions work well with a summarizer.
        This class will use an OpenAI summarizer to find the relevant strings in a text. The groundedness_provider can 
        either be an llm with OpenAI or NLI with huggingface.

        Args:
            groundedness_provider (Provider, optional): groundedness provider options: OpenAI LLM or HuggingFace NLI. Defaults to OpenAI().
        """
        if groundedness_provider is None:
            groundedness_provider = OpenAI()
        summarize_provider = OpenAI()
        if not isinstance(groundedness_provider, (OpenAI, Huggingface)):
            raise Exception(
                "Groundedness is only supported groundedness_provider as OpenAI or Huggingface Providers."
            )
        super().__init__(
            summarize_provider=summarize_provider,
            groundedness_provider=groundedness_provider,
            obj=self  # for WithClassInfo
        )

    def groundedness_measure(self, source: str, statement: str) -> float:
        """A measure to track if the source material supports each sentence in the statement. 
        This groundedness measure is faster; but less accurate than `groundedness_measure_with_summarize_step` 

        ```
        grounded = feedback.Groundedness(groundedness_provider=OpenAI())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        groundedness_scores = {}
        if isinstance(self.groundedness_provider, OpenAI):
            plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc
            if len(statement) > plausible_junk_char_min:
                reason = self.summarize_provider._groundedness_doc_in_out(
                    source, statement
                )
            i = 0
            for line in reason.split('\n'):
                if "Score" in line:
                    groundedness_scores[f"statement_{i}"
                                       ] = _re_1_10_rating(line) / 10
                    i += 1
            return groundedness_scores, {"reason": reason}
        if isinstance(self.groundedness_provider, Huggingface):
            reason = ""
            for i, hypothesis in enumerate(
                    tqdm(statement.split("."),
                         desc="Groundendess per statement in source")):
                plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc
                if len(hypothesis) > plausible_junk_char_min:
                    score = self.groundedness_provider._doc_groundedness(
                        premise=source, hypothesis=hypothesis
                    )
                    reason = reason + str.format(
                        feedback_prompts.GROUNDEDNESS_REASON_TEMPLATE,
                        statement_sentence=hypothesis,
                        supporting_evidence="[Doc NLI Used full source]",
                        score=score * 10,
                    )
                    groundedness_scores[f"statement_{i}"] = score

            return groundedness_scores, {"reason": reason}

    def groundedness_measure_with_summarize_step(
        self, source: str, statement: str
    ) -> float:
        """A measure to track if the source material supports each sentence in the statement. 
        This groundedness measure is more accurate; but slower using a two step process.
        - First find supporting evidence with an LLM
        - Then for each statement sentence, check groundendness
        ```
        grounded = feedback.Groundedness(groundedness_provider=OpenAI())


        f_groundedness = feedback.Feedback(grounded.groundedness_measure_with_summarize_step).on(
            Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
        ```
        Args:
            source (str): The source that should support the statement
            statement (str): The statement to check groundedness

        Returns:
            float: A measure between 0 and 1, where 1 means each sentence is grounded in the source.
        """
        groundedness_scores = {}
        reason = ""
        for i, hypothesis in enumerate(
                tqdm(statement.split("."),
                     desc="Groundendess per statement in source")):
            plausible_junk_char_min = 4  # very likely "sentences" under 4 characters are punctuation, spaces, etc
            if len(hypothesis) > plausible_junk_char_min:
                supporting_premise = self.summarize_provider._find_relevant_string(
                    source, hypothesis
                )
                score = self.groundedness_provider._summarized_groundedness(
                    premise=supporting_premise, hypothesis=hypothesis
                )
                reason = reason + str.format(
                    feedback_prompts.GROUNDEDNESS_REASON_TEMPLATE,
                    statement_sentence=hypothesis,
                    supporting_evidence=supporting_premise,
                    score=score * 10,
                )
                groundedness_scores[f"statement_{i}"] = score
        return groundedness_scores, {"reason": reason}

    def grounded_statements_aggregator(
        self, source_statements_matrix: np.ndarray
    ) -> float:
        """Aggregates multi-input, mulit-output information from the _groundedness_measure_experimental methods.


        Args:
            source_statements_matrix (np.ndarray): a 2D array with the first dimension corresponding to a source text,
                and the second dimension corresponding to each sentence in a statement; it's groundedness score

        Returns:
            float: for each statement, gets the max groundedness, then averages over that.
        """
        max_over_sources = np.max(source_statements_matrix, axis=0)
        return np.mean(max_over_sources)


class GroundTruthAgreement(SerialModel, WithClassInfo):
    ground_truth: Union[List[str], FunctionOrMethod]
    provider: Provider

    ground_truth_imp: Optional[Callable] = pydantic.Field(exclude=True)

    def __init__(
        self,
        ground_truth: Union[List[str], Callable, FunctionOrMethod],
        provider: Provider = None
    ):
        if provider is None:
            provider = OpenAI()
        if isinstance(ground_truth, List):
            ground_truth_imp = None
        elif isinstance(ground_truth, FunctionOrMethod):
            ground_truth_imp = ground_truth.load()
        elif isinstance(ground_truth, Callable):
            ground_truth_imp = ground_truth
            ground_truth = FunctionOrMethod.of_callable(ground_truth)
        elif isinstance(ground_truth, Dict):
            # Serialized FunctionOrMethod?
            ground_truth = FunctionOrMethod.pick(**ground_truth)
            ground_truth_imp = ground_truth.load()
        else:
            raise RuntimeError(
                f"Unhandled ground_truth type: {type(ground_truth)}."
            )

        super().__init__(
            ground_truth=ground_truth,
            ground_truth_imp=ground_truth_imp,
            provider=provider,
            obj=self  # for WithClassInfo
        )

    def _find_response(self, prompt: str) -> Optional[str]:
        if self.ground_truth_imp is not None:
            return self.ground_truth_imp(prompt)

        responses = [
            qr["response"] for qr in self.ground_truth if qr["query"] == prompt
        ]
        if responses:
            return responses[0]
        else:
            return None

    def agreement_measure(
        self, prompt: str, response: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Uses OpenAI's Chat GPT Model. A function that that measures
        similarity to ground truth. A second template is given to Chat GPT
        with a prompt that the original response is correct, and measures
        whether previous Chat GPT's response is similar.

        Parameters:
            prompt (str): A text prompt to an agent. response (str): The
            agent's response to the prompt.

        Returns:
            - float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".
            - dict: with key 'ground_truth_response'
        """
        ground_truth_response = self._find_response(prompt)
        if ground_truth_response:
            agreement_txt = self.provider._get_answer_agreement(
                prompt, response, ground_truth_response
            )
            ret = _re_1_10_rating(agreement_txt) / 10, dict(
                ground_truth_response=ground_truth_response
            )
        else:
            ret = np.nan
        return ret


class AzureOpenAI(OpenAI):
    deployment_id: str

    def __init__(self, endpoint=None, **kwargs):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Wrapper to use Azure OpenAI. Please export the following env variables

        - OPENAI_API_BASE
        - OPENAI_API_VERSION
        - OPENAI_API_KEY

        Parameters:

        - model_engine (str, optional): The specific model version. Defaults to
          "gpt-35-turbo".
        - deployment_id (str): The specified deployment id
        """

        super().__init__(
            **kwargs
        )  # need to include pydantic.BaseModel.__init__

        set_openai_key()
        openai.api_type = "azure"
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")

    def _create_chat_completion(self, *args, **kwargs):
        """
        We need to pass `engine`
        """
        return super()._create_chat_completion(
            *args, deployment_id=self.deployment_id, **kwargs
        )


# Cannot put these inside Huggingface since it interferes with pydantic.BaseModel.
HUGS_SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HUGS_TOXIC_API_URL = "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"
HUGS_CHAT_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-3B"
HUGS_LANGUAGE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"
HUGS_NLI_API_URL = "https://api-inference.huggingface.co/models/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
HUGS_DOCNLI_API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"


class Huggingface(Provider):

    endpoint: Endpoint

    def __init__(self, endpoint=None, **kwargs):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        A set of Huggingface Feedback Functions.

        All args/kwargs passed to HuggingfaceEndpoint constructor.
        """

        self_kwargs = dict()
        self_kwargs['endpoint'] = HuggingfaceEndpoint(**kwargs)

        super().__init__(
            **self_kwargs
        )  # need to include pydantic.BaseModel.__init__

    def language_match(self, text1: str, text2: str) -> float:
        """
        Uses Huggingface's papluca/xlm-roberta-base-language-detection model. A
        function that uses language detection on `text1` and `text2` and
        calculates the probit difference on the language detected on text1. The
        function is: `1.0 - (|probit_language_text1(text1) -
        probit_language_text1(text2))`
        
        Parameters:
        
            text1 (str): Text to evaluate.

            text2 (str): Comparative text to evaluate.

        Returns:

            float: A value between 0 and 1. 0 being "different languages" and 1
            being "same languages".
        """

        def get_scores(text):
            payload = {"inputs": text}
            hf_response = self.endpoint.post(
                url=HUGS_LANGUAGE_API_URL, payload=payload, timeout=30
            )
            return {r['label']: r['score'] for r in hf_response}

        max_length = 500
        scores1: AsyncResult[Dict] = TP().promise(
            get_scores, text=text1[:max_length]
        )
        scores2: AsyncResult[Dict] = TP().promise(
            get_scores, text=text2[:max_length]
        )

        scores1: Dict = scores1.get()
        scores2: Dict = scores2.get()

        langs = list(scores1.keys())
        prob1 = np.array([scores1[k] for k in langs])
        prob2 = np.array([scores2[k] for k in langs])
        diff = prob1 - prob2

        l1 = 1.0 - (np.linalg.norm(diff, ord=1)) / 2.0

        return l1, dict(text1_scores=scores1, text2_scores=scores2)

    def positive_sentiment(self, text: str) -> float:
        """
        Uses Huggingface's cardiffnlp/twitter-roberta-base-sentiment model. A
        function that uses a sentiment classifier on `text`.
        
        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "negative sentiment" and 1
            being "positive sentiment".
        """
        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}

        hf_response = self.endpoint.post(
            url=HUGS_SENTIMENT_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'LABEL_2':
                return label['score']

    def not_toxic(self, text: str) -> float:
        """
        Uses Huggingface's martin-ha/toxic-comment-model model. A function that
        uses a toxic comment classifier on `text`.
        
        Parameters:
            text (str): Text to evaluate.

        Returns:
            float: A value between 0 and 1. 0 being "toxic" and 1 being "not
            toxic".
        """
        max_length = 500
        truncated_text = text[:max_length]
        payload = {"inputs": truncated_text}
        hf_response = self.endpoint.post(
            url=HUGS_TOXIC_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'toxic':
                return label['score']

    def _summarized_groundedness(self, premise: str, hypothesis: str) -> float:
        """ A groundedness measure best used for summarized premise against simple hypothesis.
        This Huggingface implementation uses NLI.

        Args:
            premise (str): NLI Premise
            hypothesis (str): NLI Hypothesis

        Returns:
            float: NLI Entailment
        """
        if not '.' == premise[len(premise) - 1]:
            premise = premise + '.'
        nli_string = premise + ' ' + hypothesis
        payload = {"inputs": nli_string}
        hf_response = self.endpoint.post(url=HUGS_NLI_API_URL, payload=payload)

        for label in hf_response:
            if label['label'] == 'entailment':
                return label['score']

    def _doc_groundedness(self, premise, hypothesis):
        """ A groundedness measure for full document premise against hypothesis.
        This Huggingface implementation uses DocNLI. The Hypoethsis still only works on single small hypothesis.

        Args:
            premise (str): NLI Premise
            hypothesis (str): NLI Hypothesis

        Returns:
            float: NLI Entailment
        """
        nli_string = premise + ' [SEP] ' + hypothesis
        payload = {"inputs": nli_string}
        hf_response = self.endpoint.post(
            url=HUGS_DOCNLI_API_URL, payload=payload
        )

        for label in hf_response:
            if label['label'] == 'entailment':
                return label['score']


class Cohere(Provider):
    model_engine: str = "large"

    def __init__(self, model_engine='large', endpoint=None, **kwargs):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.

        kwargs['endpoint'] = Endpoint(name="cohere")
        kwargs['model_engine'] = model_engine

        super().__init__(
            **kwargs
        )  # need to include pydantic.BaseModel.__init__

    def sentiment(
        self,
        text,
    ):
        return int(
            Cohere().endpoint.run_me(
                lambda: get_cohere_agent().classify(
                    model=self.model_engine,
                    inputs=[text],
                    examples=feedback_prompts.COHERE_SENTIMENT_EXAMPLES
                )[0].prediction
            )
        )

    def not_disinformation(self, text):
        return int(
            Cohere().endpoint.run_me(
                lambda: get_cohere_agent().classify(
                    model=self.model_engine,
                    inputs=[text],
                    examples=feedback_prompts.COHERE_NOT_DISINFORMATION_EXAMPLES
                )[0].prediction
            )
        )
