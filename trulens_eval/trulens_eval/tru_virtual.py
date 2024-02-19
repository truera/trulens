"""
# Virtual Apps

Log existing results with trulens_eval and run evals against them.
"""

from concurrent import futures
import datetime
import logging
from pprint import PrettyPrinter
import time
from typing import Any, ClassVar, Dict, Optional, Union

from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import Perf
from trulens_eval.schema import Record
from trulens_eval.schema import RecordAppCall
from trulens_eval.schema import RecordAppCallMethod
from trulens_eval.schema import Select
from trulens_eval.utils.pyschema import Class
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.pyschema import Method
from trulens_eval.utils.pyschema import Module
from trulens_eval.utils.pyschema import Obj
from trulens_eval.utils.serial import GetItemOrAttribute
from trulens_eval.utils.serial import JSON
from trulens_eval.utils import serial


logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class VirtualApp(dict):
    """
    A dictionary meant to represent the components of a virtual app. `TruVirtual`
    will refer to this class as the wrapped app. All calls will be under `VirtualApp.root` 
    """

    def __setitem__(self, __name: Union[str, serial.Lens], __value: Any) -> None:
        """
        Allow setitem to work on Lenses instead of just strings. Uses `Lens.set`
        if a lens is given.
        """

        if isinstance(__name, str):
            return super().__setitem__(__name, __value)

        # Chop off __app__ or __record__ prefix if there.
        __name = Select.dequalify(__name)

        # Chop off "app" prefix if there.
        if isinstance(__name.path[0], GetItemOrAttribute) \
            and __name.path[0].get_item_or_attribute() == "app":
            __name = serial.Lens(path=__name.path[1:])

        # Does not mutate so need to use dict.update .
        temp = __name.set(self, __value)
        self.update(temp)

    def root(self):
        # All virtual calls will have this on top of the stack as if their app
        # was called using this as the main/root method.
        pass


# Create some pyschema instances to refer to things that are either in the
# virtual app.
virtual_module = Module(
    package_name="trulens_eval", module_name="trulens_eval.tru_virtual"
)
virtual_class = Class(module=virtual_module, name="VirtualApp")
virtual_object = Obj(cls=virtual_class, id=0)
virtual_method_root = Method(cls=virtual_class, obj=virtual_object, name="root")

# Method name will be replaced by the last attribute in the selector provided by
# user:
virtual_method_call = Method(
    cls=virtual_class, obj=virtual_object, name="method_name_not_set"
)


class VirtualRecord(Record):
    """Utility class for creating `Record`s using selectors.
    
    In the example below, `Select.RecordCalls.retriever` refers to a presumed
    component of some virtual model which is assumed to have called the method
    `get_context`. The inputs and outputs of that call are specified as the
    value with the selector as the key. Other than `calls`, other arguments are
    the same as for `Record`, but empty values are filled for arguments that are
    not provided but are otherwise required.

    Example:
        ```python
        VirtualRecord(
            main_input="Where is Germany?", 
            main_output="Germany is in Europe", 
            calls={
                Select.RecordCalls.retriever.get_context: {
                    'args': ["Where is Germany?"], 
                    'rets': ["Germany is a country located in Europe."]
                },
                Select.RecordCalls.some_other_component.do_something: {
                    'args': ["Some other inputs."], 
                    'rets': ["Some other output."]
                }
            }
        )
        ```
    """

    def __init__(self, calls: Dict[serial.Lens, Dict], **kwargs: dict):
        root_call = RecordAppCallMethod(path=serial.Lens(), method=virtual_method_root)

        start_time = datetime.datetime.now()

        record_calls = []

        for lens, call in calls.items():
            substart_time = datetime.datetime.now()

            # NOTE(piotrm for garrett): that the dashboard timeline has problems
            # with calls that span too little time so we add some delays to the
            # recorded perf.
            time.sleep(0.1)

            if "stack" not in call:
                path, method_name = Select.path_and_method(
                    Select.dequalify(lens)
                )
                method = virtual_method_call.replace(name=method_name)

                call['stack'] = [
                    root_call,
                    RecordAppCallMethod(path=path, method=method)
                ]
            if "args" not in call:
                call['args'] = []
            if "rets" not in call:
                call['rets'] = []
            if "pid" not in call:
                call['pid'] = 0
            if "tid" not in call:
                call['tid'] = 0

            subend_time = datetime.datetime.now()

            if "perf" not in call:
                call['perf'] = Perf(
                    start_time=substart_time, end_time=subend_time
                )

            rinfo = RecordAppCall(**call)
            record_calls.append(rinfo)

        end_time = datetime.datetime.now()

        if "cost" not in kwargs:
            kwargs['cost'] = Cost()
        if "perf" not in kwargs:
            kwargs['perf'] = Perf(start_time=start_time, end_time=end_time)

        if "main_input" not in kwargs:
            kwargs['main_input'] = "No main_input provided."
        if "main_output" not in kwargs:
            kwargs['main_output'] = "No main_output provided."

        # append root call
        record_calls.append(
            RecordAppCall(
                stack=[root_call],
                args=[kwargs['main_input']],
                rets=[kwargs['main_output']],
                perf=kwargs['perf'],
                cost=kwargs['cost'],
                tid=0,
                pid=0
            )
        )

        if "app_id" not in kwargs:
            kwargs[
                'app_id'
            ] = "No app_id provided."  # this gets replaced by TruVirtual.add_record .

        super().__init__(calls=record_calls, **kwargs)


class TruVirtual(App):
    """Recorder for virtual apps.
    
    Virtual apps are data only in that they cannot be executed but for whom
    previously-computed results can be added using
    [add_record][trulens_eval.tru_virtual.TruVirtual]. The
    [VirtualRecord][trulens_eval.tru_virtual.VirtualRecord] class may be useful
    for creating records for this. Fields used by non-virtual apps can be
    specified here, notably:

    See [App][trulens_eval.app.App] and
    [AppDefinition][trulens_eval.schema.AppDefinition] for constructor
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
            app_id="my_virtual_app",
            app=virtual_app
        )
        ```
    """

    app: VirtualApp = Field(default_factory=VirtualApp)

    root_callable: ClassVar[FunctionOrMethod] = virtual_method_root

    root_class: Any = Class.of_class(VirtualApp)

    instrument: Optional[Instrument] = None

    def __init__(self, app: Optional[Union[VirtualApp, JSON]] = None, **kwargs: dict):
        """Virtual app for logging existing app results. """

        if app is None:
            app = VirtualApp()
        else:
            if isinstance(app, dict):
                app = VirtualApp(app)
            else:
                raise ValueError(
                    "Unknown type for `app`. "
                    "Either dict or `trulens_eval.tru_virtual.VirtualApp` expected."
                )

        super().__init__(app=app, **kwargs)

    def add_record(
        self,
        record: Record,
        feedback_mode: Optional[FeedbackMode] = None
    ) -> Record:
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
            record,
            feedback_mode=feedback_mode
        )
        if record.feedback_and_future_results is not None:
            record.feedback_results = [tup[1] for tup in record.feedback_and_future_results]

        # Wait for results if mode is WITH_APP.
        if feedback_mode == FeedbackMode.WITH_APP and record.feedback_results is not None:
            futs = record.feedback_results
            futures.wait(futs)

        return record
