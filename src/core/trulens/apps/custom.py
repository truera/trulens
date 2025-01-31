"""Custom class application

This wrapper is the most flexible option for instrumenting an application, and
can be used to instrument any custom python class.

Example: Instrumenting a custom class
    Consider a mock question-answering app with a context retriever component coded
    up as two classes in two python, `CustomApp` and `CustomRetriever`:

    ### `custom_app.py`

    ```python
    from trulens.apps.custom import instrument
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
    from trulens.apps.custom import instrument

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
    from trulens.apps.custom import TruCustomApp

    custom_app = CustomApp()

    # Normal app Usage:
    response = custom_app.respond_to_query("What is the capital of Indonesia?")

    # Wrapping app with `TruCustomApp`:
    tru_recorder = TruCustomApp(ca)

    # Tracked usage:
    with tru_recorder:
        custom_app.respond_to_query, input="What is the capital of Indonesia?")
    ```

    `TruCustomApp` constructor arguments are like in those higher-level
apps as well including the feedback functions, metadata, etc.

## Instrumenting 3rd party classes

In cases you do not have access to a class to make the necessary decorations for
tracking, you can instead use one of the static methods of `instrument`, for
example, the alternative for making sure the custom retriever gets instrumented
is via:

Example:
    ```python
    # custom_app.py`:

    from trulens.apps.custom import instrument
    from some_package.from custom_retriever import CustomRetriever

    instrument.method(CustomRetriever, "retrieve_chunks")

    # ... rest of the custom class follows ...
    ```

## API Usage Tracking

Uses of python libraries for common LLMs like OpenAI are tracked in custom class
apps.

### Covered LLM Libraries

- Official OpenAI python package (https://github.com/openai/openai-python).
- Snowflake Cortex (https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex.html).
- Amazon Bedrock (https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock_code_examples.html).

### Huggingface

Uses of huggingface inference APIs are tracked as long as requests are made
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
  component hierarchy. Json-like (json bases like string, int, and containers
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
        TruCustomApp (Other) at 0x171bd3380 with path *.__app__
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
  `TruCustomApp` to provide a structure to stand in place for (or augment) the
  data produced by walking over instrumented components to make sure this
  hierarchy contains the owner of each instrumented method.

  The owner-not-found error looks like this:

```python
Function <function CustomRetriever.retrieve_chunks at 0x177935d30> was not found during instrumentation walk. Make sure it is accessible by traversing app <custom_app.CustomApp object at 0x112a005b0> or provide a bound method for it as TruCustomApp constructor argument `methods_to_instrument`.
Function <function CustomTemplate.fill at 0x1779474c0> was not found during instrumentation walk. Make sure it is accessible by traversing app <custom_app.CustomApp object at 0x112a005b0> or provide a bound method for it as TruCustomApp constructor argument `methods_to_instrument`.
Function <function CustomLLM.generate at 0x1779471f0> was not found during instrumentation walk. Make sure it is accessible by traversing app <custom_app.CustomApp object at 0x112a005b0> or provide a bound method for it as TruCustomApp constructor argument `methods_to_instrument`.
```

  Subsequent attempts at `with_record`/`awith_record` may result in the "Empty
  record" exception.

- Usage tracking not tracking. We presently have limited coverage over which
  APIs we track and make some assumptions with regards to accessible APIs
  through lower-level interfaces. Specifically, we only instrument the
  `requests` module's `post` method for the lower level tracking. Please file an
  issue on github with your use cases so we can work out a more complete
  solution as needed.
"""

import logging
from pprint import PrettyPrinter
import warnings

from trulens.apps.app import TruApp
from trulens.core import instruments as core_instruments

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# Keys used in app_extra_json to indicate an automatically added structure for
# places an instrumented method exists but no instrumented data exists
# otherwise.
PLACEHOLDER = "__tru_placeholder"


class TruCustomApp(TruApp):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TruCustomApp is being deprecated in the next major version; use TruApp instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


warnings.warn(
    """from trulens.apps.custom import instrument
        is being deprecated in the next major version; use from trulens.apps.app import instrument
        instead.""",
    DeprecationWarning,
    stacklevel=2,
)


class instrument(core_instruments.instrument):
    """
    Decorator for marking methods to be instrumented in custom classes that are
    wrapped by TruCustomApp.
    """

    @classmethod
    def method(cls, inst_cls: type, name: str) -> None:
        core_instruments.instrument.method(inst_cls, name)

        # Also make note of it for verification that it was found by the walk
        # after init.
        TruApp.functions_to_instrument.add(getattr(inst_cls, name))


TruCustomApp.model_rebuild()
