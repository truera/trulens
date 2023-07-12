"""
# Basic input output instrumentation and monitoring.
"""

from datetime import datetime
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar, Sequence

from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.provider_apis import Endpoint
from trulens_eval.schema import Cost
from trulens_eval.schema import RecordAppCall
from trulens_eval.util import Class
from trulens_eval.util import FunctionOrMethod
from trulens_eval.util import jsonify

logger = logging.getLogger(__name__)

pp = PrettyPrinter()


class TruBasicCallableInstrument(Instrument):

    class Default:
        CLASSES = lambda: {TruWrapperApp}

        # Instrument only methods with these names and of these classes.
        METHODS = {"_call": lambda o: isinstance(o, TruWrapperApp)}

    def __init__(self):
        super().__init__(
            root_method=TruBasicApp.call_with_record,
            classes=TruBasicCallableInstrument.Default.CLASSES(),
            methods=TruBasicCallableInstrument.Default.METHODS
        )


class TruWrapperApp(object):
    # the class level call (Should be immutable from the __init__)
    _call: Callable = lambda self, *args, **kwargs: self._call_fn(
        *args, **kwargs
    )

    def __init__(self, call_fn: Callable):
        self._call_fn = call_fn


class TruBasicApp(App):
    """
    A Basic app that makes little assumptions. Assumes input text and output text 
    """

    app: TruWrapperApp

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruBasicApp._call),
        const=True
    )

    def __init__(self, text_to_text: Callable, **kwargs):
        """
        Wrap a callable for monitoring.

        Arguments:
        - text_to_text: A callable string to string
        - More args in App
        - More args in AppDefinition
        - More args in WithClassInfo
        """
        assert isinstance(text_to_text("This should return a string"), str)
        super().update_forward_refs()
        app = TruWrapperApp(text_to_text)
        kwargs['app'] = TruWrapperApp(text_to_text)
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = TruBasicCallableInstrument()
        super().__init__(**kwargs)

    def call_with_record(self, input: str, **kwargs):
        """ Run the callable and pass any kwargs.

        Returns:
            dict: record metadata
        """

        # Wrapped calls will look this up by traversing the call stack. This
        # should work with threads.
        record: Sequence[RecordAppCall] = []

        ret = None
        error = None

        cost: Cost = Cost()

        start_time = None
        end_time = None

        try:
            start_time = datetime.now()
            ret, cost = Endpoint.track_all_costs_tally(
                lambda: self.app._call(input, **kwargs)
            )
            end_time = datetime.now()

        except BaseException as e:
            end_time = datetime.now()
            error = e
            logger.error(f"App raised an exception: {e}")

        assert len(record) > 0, "No information recorded in call."

        ret_record_args = dict()

        ret_record_args['main_input'] = input
        if ret is not None:
            ret_record_args['main_output'] = ret

        ret_record = self._post_record(
            ret_record_args, error, cost, start_time, end_time, record
        )

        return ret, ret_record
