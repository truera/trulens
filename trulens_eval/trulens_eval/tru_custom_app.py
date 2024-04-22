"""
# Custom class application

This wrapper is the most flexible option for instrumenting an application, and can be used to instrument any custom python class.

!!! example

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

The core tool for instrumenting these classes is the `@instrument` decorator. _TruLens_ needs to be aware
of two high-level concepts to usefully monitor the app: components and methods
used by components. The `instrument` must decorate each method that the user wishes to track.

The owner classes of any decorated method is then viewed as an app component. In this example, case `CustomApp` and
`CustomRetriever` are components. 

    !!! example

    ### `example.py`

    ```python
    from custom_app import CustomApp from trulens_eval.tru_custom_app
    import TruCustomApp

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

from inspect import signature
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Optional, Set

from pydantic import Field

from trulens_eval import app as mod_app
from trulens_eval.instruments import Instrument
from trulens_eval.instruments import instrument as base_instrument
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import Function
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.serial import Lens
from trulens_eval.utils.text import UNICODE_CHECK

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# Keys used in app_extra_json to indicate an automatically added structure for
# places an instrumented method exists but no instrumented data exists
# otherwise.
PLACEHOLDER = "__tru_placeholder"


class TruCustomApp(mod_app.App):
    """
    This recorder is the most flexible option for instrumenting an application,
    and can be used to instrument any custom python class.

    Track any custom app using methods decorated with `@instrument`, or whose
    methods are instrumented after the fact by `instrument.method`.

    !!! example "Using the `@instrument` decorator"

        ```python
        from trulens_eval import instrument
        
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

    !!! example "Using `instrument.method`"

        ```python
        from trulens_eval import instrument
        
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

    !!! example "Defining feedback functions with instrumented methods"

        ```python
        f_context_relevance = (
            Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
            .on(Select.RecordCalls.retrieve_chunks.args.query) # refers to the query arg of CustomApp's retrieve_chunks method
            .on(Select.RecordCalls.retrieve_chunks.rets.collect())
            .aggregate(np.mean)
            )
        ```

    Last, the `TruCustomApp` recorder can wrap our custom application, and
    provide logging and evaluation upon its use.

    !!! example "Using the `TruCustomApp` recorder"

        ```python
        from trulens_eval import TruCustomApp

        tru_recorder = TruCustomApp(custom_app, 
            app_id="Custom Application v1",
            feedbacks=[f_context_relevance])

        with tru_recorder as recording:
            custom_app.respond_to_query("What is the capital of Indonesia?")
        ```

        See [Feedback
        Functions](https://www.trulens.org/trulens_eval/api/feedback/) for
        instantiating feedback functions.

    Args:
        app: Any class.

        **kwargs: Additional arguments to pass to [App][trulens_eval.app.App]
            and [AppDefinition][trulens_eval.schema.app.AppDefinition]
    """

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    app: Any

    root_callable: ClassVar[FunctionOrMethod] = Field(None)

    functions_to_instrument: ClassVar[Set[Callable]] = set([])
    """Methods marked as needing instrumentation.
    
    These are checked to make sure the object walk finds them. If not, a message
    is shown to let user know how to let the TruCustomApp constructor know where
    these methods are.
    """

    main_method_loaded: Optional[Callable] = Field(None, exclude=True)
    """Main method of the custom app."""

    main_method: Optional[Function] = None
    """Serialized version of the main method."""

    def __init__(self, app: Any, methods_to_instrument=None, **kwargs: dict):
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)

        instrument = Instrument(
            app=self  # App mixes in WithInstrumentCallbacks
        )
        kwargs['instrument'] = instrument

        if 'main_method' in kwargs:
            main_method = kwargs['main_method']

            # TODO: ARGPARSE
            if isinstance(main_method, dict):
                main_method = Function.model_validate(main_method)

            if isinstance(main_method, Function):
                main_method_loaded = main_method.load()
                main_name = main_method.name

                cls = main_method.cls.load()
                mod = main_method.module.load().__name__

            else:
                main_name = main_method.__name__
                main_method_loaded = main_method
                main_method = Function.of_function(main_method_loaded)

                if not safe_hasattr(main_method_loaded, "__self__"):
                    raise ValueError(
                        "Please specify `main_method` as a bound method (like `someapp.somemethod` instead of `Someclass.somemethod`)."
                    )

                app_self = main_method_loaded.__self__

                assert app_self == app, "`main_method`'s bound self must be the same as `app`."

                cls = app_self.__class__
                mod = cls.__module__

            kwargs['main_method'] = main_method
            kwargs['main_method_loaded'] = main_method_loaded

            instrument.include_modules.add(mod)
            instrument.include_classes.add(cls)
            instrument.include_methods[main_name] = lambda o: isinstance(o, cls)

        # This does instrumentation:
        super().__init__(**kwargs)

        # Needed to split this part to after the instrumentation so that the
        # getattr below gets the instrumented version of main method.
        if 'main_method' in kwargs:
            # Set main_method to the unbound version. Will be passing in app for
            # "self" manually when needed.
            main_method_loaded = getattr(cls, main_name)

            # This will be serialized as part of this TruCustomApp. Importatly, it is unbound.
            main_method = Function.of_function(main_method_loaded, cls=cls)

            self.main_method = main_method
            self.main_method_loaded = main_method_loaded

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

            full_path = Lens().app + path

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
            obj_ids_methods_and_full_paths = list(self.get_methods_for_func(f))

            if len(obj_ids_methods_and_full_paths) == 0:
                logger.warning(
                    f"Function {f} was not found during instrumentation walk. "
                    f"Make sure it is accessible by traversing app {app} "
                    f"or provide a bound method for it as TruCustomApp constructor argument `methods_to_instrument`."
                )

            else:
                for obj_id, m, full_path in obj_ids_methods_and_full_paths:
                    try:
                        next(full_path.get(json))

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

    def main_call(self, human: str):
        if self.main_method_loaded is None:
            raise RuntimeError(
                "`main_method` was not specified so we do not know how to run this app."
            )

        sig = signature(self.main_method_loaded)
        bindings = sig.bind(self.app, human)  # self.app is app's "self"

        return self.main_method_loaded(*bindings.args, **bindings.kwargs)

    """
    # Async work ongoing:
    async def main_acall(self, human: str):
        # TODO: work in progress

        # must return an async generator of tokens/pieces that can be appended to create the full response

        if self.main_async_method is None:
            raise RuntimeError(
                "`main_async_method` was not specified so we do not know how to run this app."
            )

        sig = signature(self.main_async_method)
        bindings = sig.bind(self.app, human)  # self.app is app's "self"

        generator = await self.main_async_method(*bindings.args, **bindings.kwargs)

        return generator
    """


class instrument(base_instrument):
    """
    Decorator for marking methods to be instrumented in custom classes that are
    wrapped by TruCustomApp.
    """

    @classmethod
    def method(self_class, cls: type, name: str) -> None:
        base_instrument.method(cls, name)

        # Also make note of it for verification that it was found by the walk
        # after init.
        TruCustomApp.functions_to_instrument.add(getattr(cls, name))


import trulens_eval  # for App class annotations

TruCustomApp.model_rebuild()
