"""
# Custom class Apps

This wrapper covers apps that are not based on one of the high-level frameworks
such as langchain or llama-index. We instead assume that some python class or
classes implements an app which has similar functionality to LLM apps coded in
the high-level frameworks in that it generally processes text queries to produce
text outputs while making intermediate queries to things like LLMs, vector DBs,
and similar.

## Example Usage

Consider a mock question-answering app with a context retriever component coded
up as two classes in two python, `CustomApp` and `CustomRetriever`:

### `custom_app.py`

```python
from trulens_eval.tru_custom_app import instrument
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
from trulens_eval.tru_custom_app import instrument

class CustomRetriever:
    # NOTE: No restriction on this class either.

    @instrument
    def retrieve_chunks(self, data):
        return [
            f"Relevant chunk: {data.upper()}", f"Relevant chunk: {data[::-1]}"
        ]
```

The core tool for instrumenting these classes is the `instrument` method
(actually class, but details are not important here). trulens needs to be aware
of two high-level concepts to usefully monitor the app: components and methods
used by components. The `instrument` must decorate each method that the user
wishes to watch (for it to show up on the dashboard). In the example, all of the
functionalities are decorated. Additionally, the owner classes of any decorated
method is viewed as an app component. In this case `CustomApp` and
`CustomRetriever` are components. 

Following the instrumentation, the app can be used with or without tracking:

### `example.py`

```python
from custom_app import CustomApp from trulens_eval.tru_custom_app
import TruCustomApp

ca = CustomApp()

# Normal app usage:
response = ca.respond_to_query("What is the capital of Indonesia?")

# Wrapping app with `TruCustomApp`: 
ta = TruCustomApp(ca)

# Wrapped usage: must use the general `with_record` (or `awith_record`) method:
response, record = ta.with_record(
    ca.respond_to_query, input="What is the capital of Indonesia?"
)
```

The `with_record` use above returns both the response of the app normally
produces as well as the record of the app as is the case with the higher-level
wrappers. `TruCustomApp` constructor arguments are like in those higher-level
apps as well including the feedback functions, metadata, etc.

### Instrumenting 3rd party classes

In cases you do not have access to a class to make the necessary decorations for
tracking, you can instead use one of the static methods of `instrument`, for
example, the alterative for making sure the custom retriever gets instrumented
is via:

```python
# custom_app.py`:

from trulens_eval.tru_custom_app import instrument
from somepackage.from custom_retriever import CustomRetriever

instrument.method(CustomRetriever, "retrieve_chunks")

# ... rest of the custom class follows ...
```

## API Usage Tracking

Uses of python libraries for common LLMs like OpenAI are tracked in custom class
apps.

### Covered LLM Libraries

- Official OpenAI python package (https://github.com/openai/openai-python).

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
Custom of trulens_eval.app component: *.__app__.app
Custom of trulens_eval.app component: *.__app__.app.llm
Custom of trulens_eval.app component: *.__app__.app.retriever
Custom of trulens_eval.app component: *.__app__.app.template

Methods:
Object at 0x1064c6d30:
	<function CustomApp.retrieve_chunks at 0x14c6c5700> with path *.__app__.app
	<function CustomApp.respond_to_query at 0x14c6c5790> with path *.__app__.app
Object at 0x1064c6dc0:
	<function CustomLLM.generate at 0x14c6c51f0> with path *.__app__.app.llm
Object at 0x1064c6f70:
	<function CustomRetriever.retrieve_chunks at 0x14c6b5c10> with path *.__app__.app.retriever
Object at 0x106272100:
	<function CustomTemplate.fill at 0x14c6c54c0> with path *.__app__.app.template
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
from typing import Any, Callable, ClassVar, Set, Iterable

from pydantic import Field

from trulens_eval import Select
from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.util import Class
from trulens_eval.util import FunctionOrMethod
from trulens_eval.util import JSONPath
from trulens_eval.utils.text import UNICODE_CHECK

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# Keys used in app_extra_json to indicate an automatically added structure for
# places an instrumented method exists but no instrumented data exists
# otherwise.
PLACEHOLDER = "__tru_placeholder"


class TruCustomApp(App):
    app: Any

    root_callable: ClassVar[FunctionOrMethod] = Field(None)

    # Methods marked as needing instrumentation. These are checked to make sure
    # the object walk finds them. If not, a message is shown to let user know
    # how to let the TruCustomApp constructor know where these methods are.
    functions_to_instrument: ClassVar[Set[Callable]] = set([])

    def __init__(self, app: Any, methods_to_instrument=None, **kwargs):
        """
        Wrap a custom class for recording.

        Arguments:
        - app: Any -- the custom app object being wrapped.
        - More args in App
        - More args in AppDefinition
        - More args in WithClassInfo
        """

        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)

        kwargs['instrument'] = Instrument(
            root_methods=set(
                [TruCustomApp.with_record, TruCustomApp.awith_record]
            ),
            callbacks=self  # App mixes in WithInstrumentCallbacks
        )

        super().__init__(**kwargs)

        methods_to_instrument = methods_to_instrument or dict()

        # The rest of this code instruments methods explicitly passed to
        # constructor as needing instrumentation and checks that methods
        # decorated with @instrument or passed explicitly belong to some
        # component as per serialized version of this app. If they are not,
        # placeholders are made in `app_extra_json` so that subsequent
        # serialization looks like the components exist.
        json = self.dict()

        for m, path in methods_to_instrument.items():
            method_name = m.__name__

            full_path = JSONPath().app + path

            self.instrument.instrument_method(
                method_name=method_name, obj=m.__self__, query=full_path
            )

            # TODO: DEDUP with next condition

            # Check whether the path/location of the method is in json serialization and
            # if not, add a placeholder to app_extra_json.
            try:
                next(full_path(json))

                print(
                    f"{UNICODE_CHECK} Added method {m.__name__} under component at path {full_path}"
                )

            except Exception:
                logger.warning(
                    f"App has no component at path {full_path} . "
                    f"Specify the component with the `app_extra_json` argument to TruCustomApp constructor. "
                    f"Creating a placeholder there for now."
                )

                path.set(
                    self.app_extra_json, {
                        PLACEHOLDER:
                            "I was automatically added to `app_extra_json` because there was nothing here to refer to an instrumented method owner.",
                        m.__name__:
                            f"Placeholder for method {m.__name__}."
                    }
                )

        # Check that any functions marked with `TruCustomApp.instrument` has been
        # instrumented as a method under some object.
        for f in TruCustomApp.functions_to_instrument: 
            methods_and_full_paths = list(self._get_methods_for_func(f))

            if len(methods_and_full_paths) == 0: 
                logger.warning(
                    f"Function {f} was not found during instrumentation walk. "
                    f"Make sure it is accessible by traversing app {app} "
                    f"or provide a bound method for it as TruCustomApp constructor argument `methods_to_instrument`."
                )

            else:
                for m, full_path in methods_and_full_paths:
                    try:
                        next(full_path(json))

                    except Exception as e:
                        logger.warning(
                            f"App has no component owner of instrumented method {m} at path {full_path}. "
                            f"Specify the component with the `app_extra_json` argument to TruCustomApp constructor. "
                            f"Creating a placeholder there for now."
                        )

                        path.set(
                            self.app_extra_json, {
                                PLACEHOLDER:
                                    "I was automatically added to `app_extra_json` because there was nothing here to refer to an instrumented method owner.",
                                m.__name__:
                                    f"Placeholder for method {m.__name__}."
                            }
                        )

        # DB stuff and checks:
        self.post_init()

    def __getattr__(self, __name: str) -> Any:
        # A message for cases where a user calls something that the wrapped
        # app has but we do not wrap yet.

        if hasattr(self.app, __name):
            return RuntimeError(
                f"TruCustomApp has no attribute {__name} but the wrapped app ({type(self.app)}) does. ",
                f"If you are calling a {type(self.app)} method, retrieve it from that app instead of from `TruCustomApp`. "
            )
        else:
            raise RuntimeError(
                f"TruCustomApp nor wrapped app have attribute named {__name}."
            )


class instrument:
    """
    Decorator for marking methods to be instrumented in custom classes that are
    wrapped by TruCustomApp.
    """

    # https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class

    def __init__(self, func: Callable):
        self.func = func

    def __set_name__(self, cls: type, name: str):
        """
        For use as method decorator.
        """

        # Important: do this first:
        setattr(cls, name, self.func)

        # Note that this does not actually change the method, just adds it to
        # list of filters.
        instrument.method(cls, name)

    @staticmethod
    def method(cls: type, name: str) -> None:
        # Add the class with a method named `name`, its module, and the method
        # `name` to the Default instrumentation walk filters.
        Instrument.Default.MODULES.add(cls.__module__)
        Instrument.Default.CLASSES.add(cls)

        check_o = Instrument.Default.METHODS.get(name, lambda o: False)
        Instrument.Default.METHODS[
            name] = lambda o: check_o(o) or isinstance(o, cls)

        # Also make note of it for verification that it was found by the walk
        # after init.
        TruCustomApp.functions_to_instrument.add(getattr(cls, name))

    @staticmethod
    def methods(cls: type, names: Iterable[str]) -> None:
        for name in names:
            instrument.method(cls, name)
