"""Custom class application

This wrapper is the most flexible option for instrumenting an application, and
can be used to instrument any custom python class.

Example: Instrumenting a custom class
    Consider a mock question-answering app with a context retriever component coded
    up as two classes in two python, `CustomApp` and `CustomRetriever`:

    ### `custom_app.py`

    ```python
    from trulens.apps.app import instrument
    from custom_retriever import CustomRetriever


    class CustomApp:
        # NOTE: No restriction on this class.

        def __init__(self):
            self.retriever = CustomRetriever()

        @instrument
        def retrieve_chunks(self, data):
            return self.retriever.retrieve_chunks(data)

        @instrument
        def respond_to_query(self, input):
            chunks = self.retrieve_chunks(input) output = f"The answer to {input} is
            probably {chunks[0]} or something ..." return output
    ```

    ### `custom_retriever.py`

    ```python
    from trulens.apps.app import instrument

    class CustomRetriever:
        # NOTE: No restriction on this class either.

        @instrument
        def retrieve_chunks(self, data):
            return [
                f"Relevant chunk: {data.upper()}", f"Relevant chunk: {data[::-1]}"
            ]
    ```

The core tool for instrumenting these classes is the `@instrument` decorator.
_TruLens_ needs to be aware of two high-level concepts to usefully monitor the
app: components and methods used by components. The `instrument` must decorate
each method that the user wishes to track.

The owner classes of any decorated method is then viewed as an app component. In
this example, case `CustomApp` and `CustomRetriever` are components.

Example:
    ### `example.py`

    ```python
    from custom_app import CustomApp
    from trulens.apps.app import TruApp

    custom_app = CustomApp()

    # Normal app Usage:
    response = custom_app.respond_to_query("What is the capital of Indonesia?")

    # Wrapping app with `TruApp`:
    tru_recorder = TruApp(ca)

    # Tracked usage:
    with tru_recorder:
        custom_app.respond_to_query, input="What is the capital of Indonesia?")
    ```

    `TruApp` constructor arguments are like in those higher-level
apps as well including the feedback functions, metadata, etc.

## Instrumenting 3rd party classes

In cases you do not have access to a class to make the necessary decorations for
tracking, you can instead use one of the static methods of `instrument`, for
example, the alternative for making sure the custom retriever gets instrumented
is via:

Example:
    ```python
    # custom_app.py`:

    from trulens.apps.app import instrument
    from some_package.from custom_retriever import CustomRetriever

    instrument.method(CustomRetriever, "retrieve_chunks")

    # ... rest of the custom class follows ...
    ```

## API Usage Tracking

Uses of Python libraries for common LLMs like OpenAI are tracked in custom class
apps.

### Covered LLM Libraries

- Official OpenAI Python package (https://github.com/openai/openai-python).
- Snowflake Cortex (https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex.html).
- Amazon Bedrock (https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock_code_examples.html).

### HuggingFace

Uses of HuggingFace inference APIs are tracked as long as requests are made
through the `requests` class's `post` method to the URL
https://api-inference.huggingface.co .

## Limitations

- Tracked (instrumented) components must be accessible through other tracked
  components. Specifically, an app cannot have a custom class that is not
  instrumented but that contains an instrumented class. The inner instrumented
  class will not be found by trulens.

- All tracked components are categorized as "Custom" (as opposed to Template,
  LLM, etc.). That is, there is no categorization available for custom
  components. They will all show up as "uncategorized" in the dashboard.

- Non json-like contents of components (that themselves are not components) are
  not recorded or available in dashboard. This can be alleviated to some extent
  with the `app_extra_json` argument to `TruCustomClass` as it allows one to
  specify in the form of json additional information to store alongside the
  component hierarchy. JSON-like (JSON bases like string, int, and containers
  like sequences and dicts are included).

## What can go wrong

- If a `with_record` or `awith_record` call does not encounter any instrumented
  method, it will raise an error. You can check which methods are instrumented
  using `App.print_instrumented`. You may have forgotten to decorate relevant
  methods with `@instrument`.

```python
app.print_instrumented()

### output example:
Components:
        TruApp (Other) at 0x171bd3380 with path *.__app__
        CustomApp (Custom) at 0x12114b820 with path *.__app__.app
        CustomLLM (Custom) at 0x12114be50 with path *.__app__.app.llm
        CustomMemory (Custom) at 0x12114bf40 with path *.__app__.app.memory
        CustomRetriever (Custom) at 0x12114bd60 with path *.__app__.app.retriever
        CustomTemplate (Custom) at 0x12114bf10 with path *.__app__.app.template

Methods:
Object at 0x12114b820:
        <function CustomApp.retrieve_chunks at 0x299132ca0> with path *.__app__.app
        <function CustomApp.respond_to_query at 0x299132d30> with path *.__app__.app
        <function CustomApp.arespond_to_query at 0x299132dc0> with path *.__app__.app
Object at 0x12114be50:
        <function CustomLLM.generate at 0x299106b80> with path *.__app__.app.llm
Object at 0x12114bf40:
        <function CustomMemory.remember at 0x299132670> with path *.__app__.app.memory
Object at 0x12114bd60:
        <function CustomRetriever.retrieve_chunks at 0x299132790> with path *.__app__.app.retriever
Object at 0x12114bf10:
        <function CustomTemplate.fill at 0x299132a60> with path *.__app__.app.template
```

- If an instrumented / decorated method's owner object cannot be found when
  traversing your custom class, you will get a warning. This may be ok in the
  end but may be indicative of a problem. Specifically, note the "Tracked"
  limitation above. You can also use the `app_extra_json` argument to `App` /
  `TruApp` to provide a structure to stand in place for (or augment) the
  data produced by walking over instrumented components to make sure this
  hierarchy contains the owner of each instrumented method.

  The owner-not-found error looks like this:

```python
Function <function CustomRetriever.retrieve_chunks at 0x177935d30> was not found during instrumentation walk. Make sure it is accessible by traversing app <custom_app.CustomApp object at 0x112a005b0> or provide a bound method for it as TruApp constructor argument `methods_to_instrument`.
Function <function CustomTemplate.fill at 0x1779474c0> was not found during instrumentation walk. Make sure it is accessible by traversing app <custom_app.CustomApp object at 0x112a005b0> or provide a bound method for it as TruApp constructor argument `methods_to_instrument`.
Function <function CustomLLM.generate at 0x1779471f0> was not found during instrumentation walk. Make sure it is accessible by traversing app <custom_app.CustomApp object at 0x112a005b0> or provide a bound method for it as TruApp constructor argument `methods_to_instrument`.
```

  Subsequent attempts at `with_record`/`awith_record` may result in the "Empty
  record" exception.

- Usage tracking not tracking. We presently have limited coverage over which
  APIs we track and make some assumptions with regards to accessible APIs
  through lower-level interfaces. Specifically, we only instrument the
  `requests` module's `post` method for the lower level tracking. Please file an
  issue on GitHub with your use cases so we can work out a more complete
  solution as needed.
"""

import inspect
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Optional, Set

from pydantic import Field
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.session import TruSession
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# Keys used in app_extra_json to indicate an automatically added structure for
# places an instrumented method exists but no instrumented data exists
# otherwise.
PLACEHOLDER = "__tru_placeholder"


class TruApp(core_app.App):
    """
    This recorder is the most flexible option for instrumenting an application,
    and can be used to instrument any custom python class.

    Track any custom app using methods decorated with `@instrument`, or whose
    methods are instrumented after the fact by `instrument.method`.

    Example: Using the `@instrument` decorator
        ```python
        from trulens.core import instrument

        class CustomApp:

            def __init__(self):
                self.retriever = CustomRetriever()
                self.llm = CustomLLM()
                self.template = CustomTemplate(
                    "The answer to {question} is probably {answer} or something ..."
                )

            @instrument
            def retrieve_chunks(self, data):
                return self.retriever.retrieve_chunks(data)

            @instrument
            def respond_to_query(self, input):
                chunks = self.retrieve_chunks(input)
                answer = self.llm.generate(",".join(chunks))
                output = self.template.fill(question=input, answer=answer)

                return output

        ca = CustomApp()
        ```

    Example: Using `instrument.method`
        ```python
        from trulens.core import instrument

        class CustomApp:

            def __init__(self):
                self.retriever = CustomRetriever()
                self.llm = CustomLLM()
                self.template = CustomTemplate(
                    "The answer to {question} is probably {answer} or something ..."
                )

            def retrieve_chunks(self, data):
                return self.retriever.retrieve_chunks(data)

            def respond_to_query(self, input):
                chunks = self.retrieve_chunks(input)
                answer = self.llm.generate(",".join(chunks))
                output = self.template.fill(question=input, answer=answer)

                return output

        custom_app = CustomApp()

        instrument.method(CustomApp, "retrieve_chunks")
        ```

    Once a method is tracked, its arguments and returns are available to be used
    in feedback functions. This is done by using the `Select` class to select
    the arguments and returns of the method.

    Doing so follows the structure:

    - For args: `Select.RecordCalls.<method_name>.args.<arg_name>`

    - For returns: `Select.RecordCalls.<method_name>.rets.<ret_name>`

    Example: "Defining feedback functions with instrumented methods"

        ```python
        f_context_relevance = (
            Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
            .on(Select.RecordCalls.retrieve_chunks.args.query) # refers to the query arg of CustomApp's retrieve_chunks method
            .on(Select.RecordCalls.retrieve_chunks.rets.collect())
            .aggregate(np.mean)
            )
        ```

    Last, the `TruApp` recorder can wrap our custom application, and
    provide logging and evaluation upon its use.

    Example: Using the `TruApp` recorder
        ```python
        from trulens.apps.app import TruApp

        tru_recorder = TruApp(custom_app,
            app_name="Custom Application",
            app_version="base",
            feedbacks=[f_context_relevance])

        with tru_recorder as recording:
            custom_app.respond_to_query("What is the capital of Indonesia?")
        ```

        See [Feedback
        Functions](https://www.trulens.org/trulens/api/feedback/) for
        instantiating feedback functions.

    Args:
        app: Any class.

        **kwargs: Additional arguments to pass to [App][trulens.core.app.App]
            and [AppDefinition][trulens.core.schema.app.AppDefinition]
    """

    app: Any

    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = Field(None)

    functions_to_instrument: ClassVar[Set[Callable]] = set()
    """Methods marked as needing instrumentation.

    These are checked to make sure the object walk finds them. If not, a message
    is shown to let user know how to let the TruApp constructor know where
    these methods are.
    """

    def __init__(
        self,
        app: Any,
        main_method: Optional[Callable] = None,
        methods_to_instrument=None,
        **kwargs: Any,
    ):
        kwargs["app"] = app
        # Create `TruSession` if not already created.
        if "connector" in kwargs:
            TruSession(connector=kwargs["connector"])
        else:
            TruSession()
        if is_otel_tracing_enabled():
            main_methods = set()
            if main_method is not None:
                main_methods.add(main_method)
            for _, method in inspect.getmembers(app, inspect.ismethod):
                if self._has_record_root_instrumentation(method):
                    main_methods.add(method)
            if len(main_methods) > 1:
                raise ValueError(
                    f"Must not have more than one main method or method decorated with span type 'record_root'! Found: {list(main_methods)}"
                )
            if len(main_methods) > 0:
                main_method = main_methods.pop()

        if main_method is not None:
            kwargs["main_method"] = main_method
        kwargs["root_class"] = pyschema_utils.Class.of_object(app)

        instrument = core_instruments.Instrument(
            app=self  # App mixes in WithInstrumentCallbacks
        )
        kwargs["instrument"] = instrument

        # This does instrumentation:
        super().__init__(**kwargs)

        methods_to_instrument = methods_to_instrument or dict()

        # The rest of this code instruments methods explicitly passed to
        # constructor as needing instrumentation and checks that methods
        # decorated with @instrument or passed explicitly belong to some
        # component as per serialized version of this app. If they are not,
        # placeholders are made in `app_extra_json` so that subsequent
        # serialization looks like the components exist.
        json = self.model_dump()

        for m, path in methods_to_instrument.items():
            method_name = m.__name__

            full_path = serial_utils.Lens().app + path

            self.instrument.instrument_method(
                method_name=method_name, obj=m.__self__, query=full_path
            )

            # TODO: DEDUP with next condition

            # Check whether the path/location of the method is in json serialization and
            # if not, add a placeholder to app_extra_json.
            try:
                next(full_path(json))

                print(
                    f"{text_utils.UNICODE_CHECK} Added method {m.__name__} under component at path {full_path}"
                )

            except Exception:
                logger.warning(
                    f"App has no component at path {full_path} . "
                    f"Specify the component with the `app_extra_json` argument to TruApp constructor. "
                    f"Creating a placeholder there for now."
                )

                path.set(
                    self.app_extra_json,
                    {
                        PLACEHOLDER: "I was automatically added to `app_extra_json` because there was nothing here to refer to an instrumented method owner.",
                        m.__name__: f"Placeholder for method {m.__name__}.",
                    },
                )

        # Check that any functions marked with `TruApp.instrument` has been
        # instrumented as a method under some object.
        for f in TruApp.functions_to_instrument:
            obj_ids_methods_and_full_paths = list(self.get_methods_for_func(f))

            if len(obj_ids_methods_and_full_paths) == 0:
                logger.warning(
                    f"Function {f} was not found during instrumentation walk. "
                    f"Make sure it is accessible by traversing app {app} "
                    f"or provide a bound method for it as TruApp constructor argument `methods_to_instrument`."
                )

            else:
                for obj_id, m, full_path in obj_ids_methods_and_full_paths:
                    try:
                        next(full_path.get(json))

                    except Exception:
                        logger.debug(
                            f"App has no component owner of instrumented method {m} at path {full_path}. "
                            f"Specify the component with the `app_extra_json` argument to TruApp constructor. "
                            f"Creating a placeholder there for now."
                        )

                        full_path.set(
                            self.app_extra_json,
                            {
                                PLACEHOLDER: "I was automatically added to `app_extra_json` because there was nothing here to refer to an instrumented method owner.",
                                m.__name__: f"Placeholder for method {m.__name__}.",
                            },
                        )

    def main_call(self, human: str):
        if self.main_method_name is None:
            raise RuntimeError(
                "`main_method_name` was not specified so we do not know how to run this app."
            )

        main_method = getattr(self.app, self.main_method_name)

        return main_method(human)


class legacy_instrument(core_instruments.instrument):
    """
    Decorator for marking methods to be instrumented in custom classes that are
    wrapped by TruApp.
    """

    @classmethod
    def method(cls, inst_cls: type, name: str) -> None:
        core_instruments.instrument.method(inst_cls, name)

        # Also make note of it for verification that it was found by the walk
        # after init.
        TruApp.functions_to_instrument.add(getattr(inst_cls, name))


if is_otel_tracing_enabled():
    from trulens.core.otel.instrument import instrument as otel_instrument

    instrument = otel_instrument()
else:
    instrument = legacy_instrument

TruApp.model_rebuild()
