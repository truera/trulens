"""
Generalized root type for various libraries like llama_index and langchain .
"""

from enum import Enum
import logging
from pprint import PrettyPrinter
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from pydantic import Field

from trulens_eval.instruments import Instrument
from trulens_eval.schema import AppID
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import App
from trulens_eval.schema import Perf
from trulens_eval.schema import Query
from trulens_eval.schema import Record
from trulens_eval.tru import Tru
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import Class
from trulens_eval.util import instrumented_classes
from trulens_eval.util import json_str_of_obj
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.util import obj_id_of_obj
from trulens_eval.util import SerialModel
from trulens_eval.util import TP
from trulens_eval.util import WithClassInfo

logger = logging.getLogger(__name__)

pp = PrettyPrinter()

# App component.
COMPONENT = Any

# Component category.
# TODO: Enum
COMPONENT_CATEGORY = str


class TruApp(App, SerialModel):
    """
    Generalization of wrapped model.
    """

    # Non-serialized fields here while the serialized ones are defined in
    # `schema.py:App`.

    # Feedback functions to evaluate on each record.
    feedbacks: Sequence[Feedback] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    # NOTE: Maybe move to schema.App .
    tru: Optional[Tru] = Field(exclude=True)

    # Database interfaces for models/records/feedbacks.
    # NOTE: Maybe mobe to schema.App .
    db: Optional[TruDB] = Field(exclude=True)

    # The wrapped app.
    app: Any = Field(exclude=True)

    # Instrumentation class.
    instrument: Instrument = Field(exclude=True)

    def __init__(
        self,
        tru: Optional[Tru] = None,
        feedbacks: Optional[Sequence[Feedback]] = None,
        **kwargs
    ):

        feedbacks = feedbacks or []

        # for us:
        kwargs['tru'] = tru
        kwargs['feedbacks'] = feedbacks

        super().__init__(**kwargs)

        if tru is None:
            if self.feedback_mode != FeedbackMode.NONE:
                logger.debug("Creating default tru.")
                tru = Tru()
        else:
            if self.feedback_mode == FeedbackMode.NONE:
                logger.warn(
                    "`tru` is specified but `feedback_mode` is FeedbackMode.NONE. "
                    "No feedback evaluation and logging will occur."
                )

        self.tru = tru
        if self.tru is not None:
            self.db = tru.db

            if self.feedback_mode != FeedbackMode.NONE:
                logger.debug(
                    "Inserting app and feedback function definitions to db."
                )
                self.db.insert_app(app=self)
                for f in self.feedbacks:
                    self.db.insert_feedback_definition(f)

        else:
            if len(feedbacks) > 0:
                raise ValueError(
                    "Feedback logging requires `tru` to be specified."
                )

        self.instrument.instrument_object(
            obj=self.app, query=Query.Query().model
        )

    def json(self, *args, **kwargs):
        # Need custom jsonification here because it is likely the model
        # structure contains loops.

        return json_str_of_obj(self.dict(), *args, **kwargs)

    def dict(self):
        # Same problem as in json.
        return jsonify(self, instrument=self.instrument)

    def _post_record(
        self, ret_record_args, error, total_tokens, total_cost, start_time,
        end_time, record
    ):
        """
        Final steps of record construction common among model types.
        """

        ret_record_args['main_error'] = str(error)
        ret_record_args['calls'] = record
        ret_record_args['cost'] = Cost(n_tokens=total_tokens, cost=total_cost)
        ret_record_args['perf'] = Perf(start_time=start_time, end_time=end_time)
        ret_record_args['app_id'] = self.app_id

        ret_record = Record(**ret_record_args)

        if error is not None:
            if self.feedback_mode == FeedbackMode.WITH_APP:
                self._handle_error(record=ret_record, error=error)

            elif self.feedback_mode in [FeedbackMode.DEFERRED,
                                        FeedbackMode.WITH_APP_THREAD]:
                TP().runlater(
                    self._handle_error, record=ret_record, error=error
                )

            raise error

        if self.feedback_mode == FeedbackMode.WITH_APP:
            self._handle_record(record=ret_record)

        elif self.feedback_mode in [FeedbackMode.DEFERRED,
                                    FeedbackMode.WITH_APP_THREAD]:
            TP().runlater(self._handle_record, record=ret_record)

        return ret_record

    def _handle_record(self, record: Record):
        """
        Write out record-related info to database if set.
        """

        if self.tru is None or self.feedback_mode is None:
            return

        record_id = self.tru.add_record(record=record)

        if len(self.feedbacks) == 0:
            return

        # Add empty (to run) feedback to db.
        if self.feedback_mode == FeedbackMode.DEFERRED:
            for f in self.feedbacks:
                self.db.insert_feedback(
                    FeedbackResult(
                        name=f.name,
                        record_id=record_id,
                        feedback_definition_id=f.feedback_definition_id
                    )
                )

        elif self.feedback_mode in [FeedbackMode.WITH_APP,
                                    FeedbackMode.WITH_APP_THREAD]:

            results = self.tru.run_feedback_functions(
                record=record, feedback_functions=self.feedbacks, app=self
            )

            for result in results:
                self.tru.add_feedback(result)

    def _handle_error(self, record: Record, error: Exception):
        if self.db is None:
            return

    def instrumented(
        self, categorizer: Callable[[Class], Iterable[COMPONENT_CATEGORY]]
    ) -> Iterable[Tuple[JSONPath, List[COMPONENT_CATEGORY]]]:
        # Enumerate instrumented components:

        from trulens_eval.utils.langchain import Is

        for q, ci, obj in instrumented_classes(jsonify(
                self.app, instrument=self.instrument)):
            yield (q, list(categorizer(ci)))
