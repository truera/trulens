"""
# Virtual Apps

This module facilitates the ingestion and evaluation of application logs that
were generated outside of TruLens. It allows for the creation of a virtual
representation of your application, enabling the evaluation of logged data
within the TruLens framework.

To begin, construct a virtual application representation. This can be
achieved through a simple dictionary or by utilizing the `VirtualApp` class,
which allows for a more structured approach to storing application
information relevant for feedback evaluation.

Example: Constructing a Virtual Application
    ```python
    virtual_app = {
        'llm': {'modelname': 'some llm component model name'},
        'template': 'information about the template used in the app',
        'debug': 'optional fields for additional debugging information'
    }
    # Converting the dictionary to a VirtualApp instance
    from trulens.core import Select
    from trulens.apps.virtual import VirtualApp

    virtual_app = VirtualApp(virtual_app)
    virtual_app[Select.RecordCalls.llm.maxtokens] = 1024
    ```

Incorporate components into the virtual app for evaluation by utilizing the
`Select` class. This approach allows for the reuse of setup configurations
when defining feedback functions.

Example: Incorporating Components into the Virtual App
    ```python
    # Setting up a virtual app with a retriever component
    from trulens.core import Select
    retriever_component = Select.RecordCalls.retriever
    virtual_app[retriever_component] = 'this is the retriever component'
    ```

With your virtual app configured, it's ready to store logged data.
`VirtualRecord` offers a structured way to build records from your data for
ingestion into TruLens, distinguishing itself from direct `Record` creation
by specifying calls through selectors.

Below is an example of adding records for a context retrieval component,
emphasizing that only the data intended for tracking or evaluation needs to
be provided.

Example: Adding Records for a Context Retrieval Component
    ```python
    from trulens.apps.virtual import VirtualRecord

    # Selector for the context retrieval component's `get_context` call
    context_call = retriever_component.get_context

    # Creating virtual records
    rec1 = VirtualRecord(
        main_input='Where is Germany?',
        main_output='Germany is in Europe',
        calls={
            context_call: {
                'args': ['Where is Germany?'],
                'rets': ['Germany is a country located in Europe.']
            }
        }
    )
    rec2 = VirtualRecord(
        main_input='Where is Germany?',
        main_output='Poland is in Europe',
        calls={
            context_call: {
                'args': ['Where is Germany?'],
                'rets': ['Poland is a country located in Europe.']
            }
        }
    )

    data = [rec1, rec2]
    ```

For existing datasets, such as a dataframe of prompts, contexts, and
responses, iterate through the dataframe to create virtual records for each
entry.

Example: Creating Virtual Records from a DataFrame
    ```python
    import pandas as pd

    # Example dataframe
    data = {
        'prompt': ['Where is Germany?', 'What is the capital of France?'],
        'response': ['Germany is in Europe', 'The capital of France is Paris'],
        'context': [
            'Germany is a country located in Europe.',
            'France is a country in Europe and its capital is Paris.'
        ]
    }
    df = pd.DataFrame(data)

    # Ingesting data from the dataframe into virtual records
    data_dict = df.to_dict('records')
    data = []

    for record in data_dict:
        rec = VirtualRecord(
            main_input=record['prompt'],
            main_output=record['response'],
            calls={
                context_call: {
                    'args': [record['prompt']],
                    'rets': [record['context']]
                }
            }
        )
        data.append(rec)
    ```

After constructing the virtual records, feedback functions can be developed
in the same manner as with non-virtual applications, using the newly added
`context_call` selector for reference. The same process can be repeated for
any additional selector you add.

Example: Developing Feedback Functions
    ```python
    from trulens.providers.openai import OpenAI
    from trulens.core.feedback.feedback import Feedback

    # Initializing the feedback provider
    openai = OpenAI()

    # Defining the context for feedback using the virtual `get_context` call
    context = context_call.rets[:]

    # Creating a feedback function for context relevance
    f_context_relevance = Feedback(openai.context_relevance).on_input().on(context)
    ```

These feedback functions are then integrated into `TruVirtual` to construct
the recorder, which can handle most configurations applicable to non-virtual
apps.

Example: Integrating Feedback Functions into TruVirtual
    ```python
    from trulens.apps.virtual import TruVirtual

    # Setting up the virtual recorder
    virtual_recorder = TruVirtual(
        app_name='a virtual app',
        app_version='base',
        app=virtual_app,
        feedbacks=[f_context_relevance]
    )
    ```

To process the records and run any feedback functions associated with the
recorder, use the `add_record` method.

Example: "Logging records and running feedback functions"

    ```python
    # Ingesting records into the virtual recorder
    for record in data:
        virtual_recorder.add_record(record)
    ```

Metadata about your application can also be included in the `VirtualApp` for
evaluation purposes, offering a flexible way to store additional information
about the components of an LLM app.

Example: Storing metadata in a VirtualApp
    ```python
    # Example of storing metadata in a VirtualApp
    virtual_app = {
        'llm': {'modelname': 'some llm component model name'},
        'template': 'information about the template used in the app',
        'debug': 'optional debugging information'
    }

    from trulens.core import Select
    from trulens.apps.virtual import VirtualApp

    virtual_app = VirtualApp(virtual_app)
    virtual_app[Select.RecordCalls.llm.maxtokens] = 1024
    ```

This approach is particularly beneficial for evaluating the components of an LLM app.

Example: Evaluating components of an LLM application
    ```python
    # Adding a retriever component to the virtual app
    retriever_component = Select.RecordCalls.retriever
    virtual_app[retriever_component] = 'this is the retriever component'
    ```
"""

from concurrent import futures
import datetime
import logging
from pprint import PrettyPrinter
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Union

from pydantic import Field
from trulens.core import app as core_app
from trulens.core import instruments as core_instruments
from trulens.core.schema import base as base_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import select as select_schema
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class VirtualApp(dict):
    """A dictionary meant to represent the components of a virtual app.

    `TruVirtual` will refer to this class as the wrapped app. All calls will be
    under `VirtualApp.root`
    """

    @classmethod
    def select_context(cls):
        """Select the context of the virtual app. This is fixed to return the default path."""
        return select_schema.Select.RecordCalls.retriever.get_context.rets[:]

    def __setitem__(
        self, __name: Union[str, serial_utils.Lens], __value: Any
    ) -> None:
        """
        Allow setitem to work on Lenses instead of just strings. Uses `Lens.set`
        if a lens is given.
        """

        if isinstance(__name, str):
            return super().__setitem__(__name, __value)

        # Chop off __app__ or __record__ prefix if there.
        __name = select_schema.Select.dequalify(__name)

        # Chop off "app" prefix if there.
        if (
            isinstance(__name.path[0], serial_utils.GetItemOrAttribute)
            and __name.path[0].get_item_or_attribute() == "app"
        ):
            __name = serial_utils.Lens(path=__name.path[1:])

        # Does not mutate so need to use dict.update .
        temp = __name.set(self, __value)
        self.update(temp)

    def root(self):
        """All virtual calls will have this on top of the stack as if their app
        was called using this as the main/root method."""

        pass


virtual_module = pyschema_utils.Module(
    package_name="trulens", module_name="trulens.apps.virtual"
)
"""Module to represent the module of virtual apps.

Virtual apps will record this as their module.
"""

virtual_class = pyschema_utils.Class(module=virtual_module, name="VirtualApp")
"""Class to represent the class of virtual apps.

Virtual apps will record this as their class.
"""

virtual_object = pyschema_utils.Obj(cls=virtual_class, id=0)
"""Object to represent instances of virtual apps.

Virtual apps will record this as their instance.
"""

virtual_method_root = pyschema_utils.Method(
    cls=virtual_class, obj=virtual_object, name="root"
)
"""Method call to represent the root call of virtual apps.

Virtual apps will record this as their root call.
"""

virtual_method_call = pyschema_utils.Method(
    cls=virtual_class, obj=virtual_object, name="method_name_not_set"
)
"""Method call to represent virtual app calls that do not provide this
information.

Method name will be replaced by the last attribute in the selector provided by user.
"""


class VirtualRecord(record_schema.Record):
    """Virtual records for virtual apps.

    Many arguments are filled in by default values if not provided. See
    [Record][trulens.core.schema.record.Record] for all arguments. Listing here is
    only for those which are required for this method or filled with default values.

    Args:
        calls: A dictionary of calls to be recorded. The keys are selectors
            and the values are dictionaries with the keys listed in the next
            section.

        cost: Defaults to zero cost.

        perf: Defaults to time spanning the processing of this virtual
            record. Note that individual calls also include perf. Time span
            is extended to make sure it is not of duration zero.

    Call values are dictionaries containing arguments to
    [RecordAppCall][trulens.core.schema.record.RecordAppCall] constructor. Values
    can also be lists of the same. This happens in non-virtual apps when the
    same method is recorded making multiple calls in a single app
    invocation. The following defaults are used if not provided.

    | PARAMETER | TYPE |DEFAULT |
    | --- | ---| --- |
    | `stack` | [List][typing.List][[RecordAppCallMethod][trulens.core.schema.record.RecordAppCallMethod]] | Two frames: a root call followed by a call by [virtual_object][trulens.apps.virtual.virtual_object], method name derived from the last element of the selector of this call. |
    | `args` | [JSON][trulens.core.utils.serial.JSON] | `[]` |
    | `rets` | [JSON][trulens.core.utils.serial.JSON] | `[]` |
    | `perf` | [Perf][trulens.core.schema.base.Perf] | Time spanning the processing of this virtual call. |
    | `pid` | [int][] | `0` |
    | `tid` | [int][] | `0` |
    """

    def __init__(
        self,
        calls: Dict[serial_utils.Lens, Union[Dict, Sequence[Dict]]],
        cost: Optional[base_schema.Cost] = None,
        perf: Optional[base_schema.Perf] = None,
        **kwargs: Any,
    ):
        root_call = record_schema.RecordAppCallMethod(
            path=serial_utils.Lens(), method=virtual_method_root
        )

        record_calls = []

        start_time = datetime.datetime.now()

        for lens, call_or_calls in calls.items():
            if isinstance(call_or_calls, Sequence):
                calls_list = call_or_calls
            else:
                calls_list = [call_or_calls]

            for call in calls_list:
                substart_time = datetime.datetime.now()

                if "stack" not in call:
                    path, method_name = select_schema.Select.path_and_method(
                        select_schema.Select.dequalify(lens)
                    )
                    method = virtual_method_call.replace(name=method_name)

                    call["stack"] = [
                        root_call,
                        record_schema.RecordAppCallMethod(
                            path=path, method=method
                        ),
                    ]

                if "args" not in call:
                    call["args"] = []
                if "rets" not in call:
                    call["rets"] = []
                if "pid" not in call:
                    call["pid"] = 0
                if "tid" not in call:
                    call["tid"] = 0

                subend_time = datetime.datetime.now()

                # NOTE(piotrm for garrett): that the dashboard timeline has problems
                # with calls that span too little time so we add some delays to the
                # recorded perf.
                if (subend_time - substart_time).total_seconds() == 0.0:
                    subend_time += datetime.timedelta(microseconds=1)

                if "perf" not in call:
                    call["perf"] = base_schema.Perf(
                        start_time=substart_time, end_time=subend_time
                    )

                rinfo = record_schema.RecordAppCall(**call)
                record_calls.append(rinfo)

        end_time = datetime.datetime.now()

        # NOTE(piotrm for garrett): that the dashboard timeline has problems
        # with calls that span too little time so we add some delays to the
        # recorded perf.
        if (end_time - start_time).total_seconds() == 0.0:
            end_time += datetime.timedelta(microseconds=1)

        kwargs["cost"] = cost or base_schema.Cost()
        kwargs["perf"] = perf or base_schema.Perf(
            start_time=start_time, end_time=end_time
        )

        if "main_input" not in kwargs:
            kwargs["main_input"] = "No main_input provided."
        if "main_output" not in kwargs:
            kwargs["main_output"] = "No main_output provided."

        # append root call
        record_calls.append(
            record_schema.RecordAppCall(
                stack=[root_call],
                args=[kwargs["main_input"]],
                rets=[kwargs["main_output"]],
                perf=kwargs["perf"],
                cost=kwargs["cost"],
                tid=0,
                pid=0,
            )
        )

        if "app_id" not in kwargs:
            kwargs["app_id"] = (
                "No app_id provided."  # this gets replaced by TruVirtual.add_record .
            )

        super().__init__(calls=record_calls, **kwargs)


class TruVirtual(core_app.App):
    """Recorder for virtual apps.

    Virtual apps are data only in that they cannot be executed but for whom
    previously-computed results can be added using
    [add_record][trulens.apps.virtual.TruVirtual]. The
    [VirtualRecord][trulens.apps.virtual.VirtualRecord] class may be useful
    for creating records for this. Fields used by non-virtual apps can be
    specified here, notably:

    See [App][trulens.core.app.App] and
    [AppDefinition][trulens.core.schema.app.AppDefinition] for constructor
    arguments.

    # The `app` field.

    You can store any information you would like by passing in a dictionary to
    TruVirtual in the `app` field. This may involve an index of components or
    versions, or anything else. You can refer to these values for evaluating
    feedback.

    Usage:
        You can use `VirtualApp` to create the `app` structure or a plain
        dictionary. Using `VirtualApp` lets you use Selectors to define components:

        ```python
        virtual_app = VirtualApp()
        virtual_app[Select.RecordCalls.llm.maxtokens] = 1024
        ```

    Example:
        ```python
        virtual_app = dict(
            llm=dict(
                modelname="some llm component model name"
            ),
            template="information about the template I used in my app",
            debug="all of these fields are completely optional"
        )

        virtual = TruVirtual(
            app_name="my_virtual_app",
            app_version="base",
            app=virtual_app
        )
        ```
    """

    app: VirtualApp = Field(default_factory=VirtualApp)

    root_callable: ClassVar[pyschema_utils.FunctionOrMethod] = (
        virtual_method_root
    )

    root_class: Any = pyschema_utils.Class.of_class(VirtualApp)

    instrument: Optional[core_instruments.Instrument] = None

    selector_check_warning: bool = False
    """Selector checking is disabled for virtual apps."""

    selector_nocheck: bool = True
    """The selector check must be disabled for virtual apps.

    This is because methods that could be called are not known in advance of
    creating virtual records.
    """

    def __init__(
        self,
        app: Optional[Union[VirtualApp, serial_utils.JSON]] = None,
        **kwargs: Any,
    ):
        """Virtual app for logging existing app results."""

        if app is None:
            app = VirtualApp()
        else:
            if isinstance(app, dict):
                app = VirtualApp(app)
            else:
                raise ValueError(
                    "Unknown type for `app`. "
                    "Either dict or `trulens.apps.virtual.VirtualApp` expected."
                )

        if (
            kwargs.get("selector_nocheck") is False
            or kwargs.get("selector_check_warning") is True
        ):
            raise ValueError(
                "Selector prechecking does not work with virtual apps. "
                "The settings `selector_nocheck=True` and `selector_check_warning=False` are required."
            )

        super().__init__(app=app, **kwargs)

    def add_record(
        self,
        record: record_schema.Record,
        feedback_mode: Optional[feedback_schema.FeedbackMode] = None,
    ) -> record_schema.Record:
        """Add the given record to the database and evaluate any pre-specified
        feedbacks on it.

        The class `VirtualRecord` may be useful for creating
        records for virtual models. If `feedback_mode` is specified, will use
        that mode for this record only.
        """

        if feedback_mode is None:
            feedback_mode = self.feedback_mode

        record.app_id = self.app_id

        # Creates feedback futures.
        record.feedback_and_future_results = self._handle_record(
            record, feedback_mode=feedback_mode
        )
        if record.feedback_and_future_results is not None:
            record.feedback_results = [
                tup[1] for tup in record.feedback_and_future_results
            ]

        # Wait for results if mode is WITH_APP.
        if (
            feedback_mode == feedback_schema.FeedbackMode.WITH_APP
            and record.feedback_results is not None
        ):
            futs = record.feedback_results
            futures.wait(futs)

        return record

    def add_dataframe(
        self,
        df,
        feedback_mode: Optional[feedback_schema.FeedbackMode] = None,
    ) -> List[record_schema.Record]:
        """Add the given dataframe as records to the database and evaluate any pre-specified
        feedbacks on them.

        The class `VirtualRecord` may be useful for creating records for virtual models.

        If `feedback_mode` is specified, will use that mode for these records only.
        """

        data_dict = df.to_dict("records")
        records = []

        for record in data_dict:
            if "contexts" in record:
                retriever = select_schema.Select.RecordCalls.retriever
                if retriever not in self.app:
                    self.app[retriever] = None
                self.app[retriever] = record["contexts"]

                context_rets = (
                    record["contexts"]
                    if isinstance(record["contexts"], list)
                    else [record["contexts"]]
                )

                rec = VirtualRecord(
                    calls={
                        retriever.get_context: dict(
                            args=[record["query"]], rets=context_rets
                        )
                    },
                    main_input=record["query"],
                    main_output=record["response"],
                )
            else:
                rec = VirtualRecord(
                    main_input=record["query"],
                    main_output=record["response"],
                    calls={},
                )
            records.append(rec)

        for record in records:
            self.add_record(record, feedback_mode=feedback_mode)

        return records


TruVirtual.model_rebuild()
