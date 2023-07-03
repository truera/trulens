"""
# Langchain instrumentation and monitoring.
"""

from datetime import datetime
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, ClassVar,Sequence

from pydantic import Field

from trulens_eval.app import App
from trulens_eval.provider_apis import Endpoint
from trulens_eval.schema import Cost
from trulens_eval.schema import RecordAppCall
from trulens_eval.instruments import Instrument
from trulens_eval.util import FunctionOrMethod
from trulens_eval.util import jsonify

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

class NoInstrument(Instrument):
    pass
class NoRootClass:
    pass

class TruBasicApp(App):
    """
    A Basic app that makes little assumptions. Assumes input text and output text 
    """

    app: Callable

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(TruBasicApp._call),
        const=True
    )

    # Normally pydantic does not like positional args but chain here is
    # important enough to make an exception.
    def __init__(self, text_to_text: Callable, **kwargs):
        """
        Wrap a callable for monitoring.

        Arguments:
        - app: Chain -- the chain to wrap.
        - More args in App
        - More args in AppDefinition
        - More args in WithClassInfo
        """

        super().update_forward_refs()
        kwargs['app'] = text_to_text
        kwargs['root_class'] = NoRootClass
        kwargs['instrument'] = NoInstrument
        super().__init__(**kwargs)

    # NOTE: Input signature compatible with langchain.chains.base.Chain.__call__
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
                lambda: self.app(input, **kwargs)
            )
            end_time = datetime.now()

        except BaseException as e:
            end_time = datetime.now()
            error = e
            logger.error(f"App raised an exception: {e}")

        assert len(record) > 0, "No information recorded in call."

        ret_record_args = dict()

        inputs = self.app.prep_inputs(inputs)

        output_key = self.output_keys[0]

        ret_record_args['main_input'] = jsonify(input)

        if ret is not None:
            ret_record_args['main_output'] = jsonify(ret[output_key])

        if error is not None:
            ret_record_args['main_error'] = jsonify(error)

        ret_record = self._post_record(
            ret_record_args, error, cost, start_time, end_time, record
        )

        return ret, ret_record
