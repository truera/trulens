"""Serializable feedback-related classes."""

from __future__ import annotations

import datetime
from enum import Enum
import logging
from pprint import pformat
from typing import (
    Any, ClassVar, Dict, Hashable, List, Optional, Tuple, TypeVar, Union
)

import pydantic

from trulens_eval import app as mod_app
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import types as mod_types_schema
from trulens_eval.utils import pyschema
from trulens_eval.utils import serial
from trulens_eval.utils.json import obj_id_of_obj
from trulens_eval.utils.text import retab

T = TypeVar("T")

logger = logging.getLogger(__name__)


class Select:
    """
    Utilities for creating selectors using Lens and aliases/shortcuts.
    """

    # TODEP
    Query = serial.Lens
    """Selector type."""

    Tru: serial.Lens = Query()
    """Selector for the tru wrapper (TruLlama, TruChain, etc.)."""

    Record: Query = Query().__record__
    """Selector for the record."""

    App: Query = Query().__app__
    """Selector for the app."""

    RecordInput: Query = Record.main_input
    """Selector for the main app input."""

    RecordOutput: Query = Record.main_output
    """Selector for the main app output."""

    RecordCalls: Query = Record.app  # type: ignore
    """Selector for the calls made by the wrapped app.
    
    Layed out by path into components.
    """

    RecordCall: Query = Record.calls[-1]
    """Selector for the first called method (last to return)."""

    RecordArgs: Query = RecordCall.args
    """Selector for the whole set of inputs/arguments to the first called / last method call."""

    RecordRets: Query = RecordCall.rets
    """Selector for the whole output of the first called / last returned method call."""

    @staticmethod
    def path_and_method(select: Select.Query) -> Tuple[Select.Query, str]:
        """
        If `select` names in method as the last attribute, extract the method name
        and the selector without the final method name.
        """

        if len(select.path) == 0:
            raise ValueError(
                "Given selector is empty so does not name a method."
            )

        firsts = select.path[:-1]
        last = select.path[-1]

        if not isinstance(last, serial.StepItemOrAttribute):
            raise ValueError(
                "Last part of selector is not an attribute so does not name a method."
            )

        method_name = last.get_item_or_attribute()
        path = Select.Query(path=firsts)

        return path, method_name

    @staticmethod
    def dequalify(select: Select.Query) -> Select.Query:
        """If the given selector qualifies record or app, remove that qualification."""

        if len(select.path) == 0:
            return select

        if select.path[0] == Select.Record.path[0] or \
            select.path[0] == Select.App.path[0]:
            return Select.Query(path=select.path[1:])

        return select

    @staticmethod
    def context(app: Optional[Any] = None) -> serial.Lens:
        return mod_app.App.select_context(app)

    @staticmethod
    def for_record(query: Select.Query) -> Query:
        return Select.Query(path=Select.Record.path + query.path)

    @staticmethod
    def for_app(query: Select.Query) -> Query:
        return Select.Query(path=Select.App.path + query.path)

    @staticmethod
    def render_for_dashboard(query: Select.Query) -> str:
        """Render the given query for use in dashboard to help user specify feedback functions."""

        if len(query) == 0:
            return "Select.Query()"

        ret = ""
        rest = None

        if query.path[0:2] == Select.RecordInput.path:
            ret = "Select.RecordInput"
            rest = query.path[2:]
        elif query.path[0:2] == Select.RecordOutput.path:
            ret = "Select.RecordOutput"
            rest = query.path[2:]

        elif query.path[0:4] == Select.RecordArgs.path:
            ret = "Select.RecordArgs"
            rest = query.path[4:]
        elif query.path[0:4] == Select.RecordRets.path:
            ret = "Select.RecordRets"
            rest = query.path[4:]

        elif query.path[0:2] == Select.RecordCalls.path:
            ret = "Select.RecordCalls"
            rest = query.path[2:]

        elif query.path[0:3] == Select.RecordCall.path:
            ret = "Select.RecordCall"
            rest = query.path[3:]

        elif query.path[0] == Select.Record.path[0]:
            ret = "Select.Record"
            rest = query.path[1:]
        elif query.path[0] == Select.App.path[0]:
            ret = "Select.App"
            rest = query.path[1:]
        else:
            rest = query.path

        for step in rest:
            ret += repr(step)

        return f"{ret}"


class FeedbackMode(str, Enum):
    """Mode of feedback evaluation.

    Specify this using the `feedback_mode` to [App][trulens_eval.app.App] constructors.
    """

    NONE = "none"
    """No evaluation will happen even if feedback functions are specified."""

    WITH_APP = "with_app"
    """Try to run feedback functions immediately and before app returns a
    record."""

    WITH_APP_THREAD = "with_app_thread"
    """Try to run feedback functions in the same process as the app but after
    it produces a record."""

    DEFERRED = "deferred"
    """Evaluate later via the process started by
    `tru.start_deferred_feedback_evaluator`."""


class FeedbackResultStatus(Enum):
    """For deferred feedback evaluation, these values indicate status of evaluation."""

    NONE = "none"
    """Initial value is none."""

    RUNNING = "running"
    """Once queued/started, status is updated to "running"."""

    FAILED = "failed"
    """Run failed."""

    DONE = "done"
    """Run completed successfully."""

    SKIPPED = "skipped"
    """This feedback was skipped.
     
    This can be because because it had an `if_exists` selector and did not
    select anything or it has a selector that did not select anything the
    `on_missing` was set to warn or ignore.
    """


class FeedbackOnMissingParameters(str, Enum):
    """How to handle missing parameters in feedback function calls.
    
    This is specifically for the case were a feedback function has a selector
    that selects something that does not exist in a record/app.
    """

    ERROR = "error"
    """Raise an error if a parameter is missing.
    
    The result status will be set to
    [FAILED][trulens_eval.schema.feedback.FeedbackResultStatus.FAILED].
    """

    WARN = "warn"
    """Warn if a parameter is missing.
    
    The result status will be set to
    [SKIPPED][trulens_eval.schema.feedback.FeedbackResultStatus.SKIPPED].
    """

    IGNORE = "ignore"
    """Do nothing. 
    
    No warning or error message will be shown. The result status will be set to
    [SKIPPED][trulens_eval.schema.feedback.FeedbackResultStatus.SKIPPED].
    """


class FeedbackCall(serial.SerialModel):
    """Invocations of feedback function results in one of these instances.
    
    Note that a single `Feedback` instance might require more than one call.
    """

    args: Dict[str, Optional[serial.JSON]]
    """Arguments to the feedback function."""

    ret: float
    """Return value."""

    meta: Dict[str, Any] = pydantic.Field(default_factory=dict)
    """Any additional data a feedback function returns to display alongside its float result."""

    def __str__(self) -> str:
        out = ""
        tab = "  "
        for k, v in self.args.items():
            out += f"{tab}{k} = {v}\n"
        out += f"{tab}ret = {self.ret}\n"
        if self.meta:
            out += f"{tab}meta = \n{retab(tab=tab*2, s=pformat(self.meta))}\n"

        return out

    def __repr__(self) -> str:
        return str(self)


class FeedbackResult(serial.SerialModel):
    """Feedback results for a single [Feedback][trulens_eval.feedback.feedback.Feedback] instance.
    
    This might involve multiple feedback function calls. Typically you should
    not be constructing these objects yourself except for the cases where you'd
    like to log human feedback.

    Attributes:
        feedback_result_id (str): Unique identifier for this result.

        record_id (str): Record over which the feedback was evaluated.

        feedback_definition_id (str): The id of the
            [FeedbackDefinition][trulens_eval.schema.feedback.FeedbackDefinition] which
            was evaluated to get this result.

        last_ts (datetime.datetime): Last timestamp involved in the evaluation.

        status (FeedbackResultStatus): For deferred feedback evaluation, the
            status of the evaluation.

        cost (Cost): Cost of the evaluation.

        name (str): Given name of the feedback.

        calls (List[FeedbackCall]): Individual feedback function invocations.

        result (float): Final result, potentially aggregating multiple calls.

        error (str): Error information if there was an error.

        multi_result (str): TODO: doc
    """

    feedback_result_id: mod_types_schema.FeedbackResultID

    # Record over which the feedback was evaluated.
    record_id: mod_types_schema.RecordID

    # The `Feedback` / `FeedbackDefinition` which was evaluated to get this
    # result.
    feedback_definition_id: Optional[mod_types_schema.FeedbackDefinitionID
                                    ] = None

    # Last timestamp involved in the evaluation.
    last_ts: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now
    )

    status: FeedbackResultStatus = FeedbackResultStatus.NONE
    """For deferred feedback evaluation, the status of the evaluation."""

    cost: mod_base_schema.Cost = pydantic.Field(
        default_factory=mod_base_schema.Cost
    )

    # Given name of the feedback.
    name: str

    # Individual feedback function invocations.
    calls: List[FeedbackCall] = []

    # Final result, potentially aggregating multiple calls.
    result: Optional[float] = None

    # Error information if there was an error.
    error: Optional[str] = None

    # TODO: doc
    multi_result: Optional[str] = None

    def __init__(
        self,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None,
        **kwargs
    ):
        super().__init__(feedback_result_id="temporary", **kwargs)

        if feedback_result_id is None:
            feedback_result_id = obj_id_of_obj(
                self.model_dump(), prefix="feedback_result"
            )

        self.feedback_result_id = feedback_result_id

    def __str__(self):
        out = f"{self.name} ({self.status}) = {self.result}\n"
        for call in self.calls:
            out += pformat(call)

        return out

    def __repr__(self):
        return str(self)


class FeedbackCombinations(str, Enum):
    """How to collect arguments for feedback function calls.
    
    Note that this applies only to cases where selectors pick out more than one
    thing for feedback function arguments. This option is used for the field
    `combinations` of
    [FeedbackDefinition][trulens_eval.schema.feedback.FeedbackDefinition] and can be
    specified with
    [Feedback.aggregate][trulens_eval.feedback.feedback.Feedback.aggregate].
    """

    ZIP = "zip"
    """Match argument values per position in produced values. 
    
    Example:
        If the selector for `arg1` generates values `0, 1, 2` and one for `arg2`
        generates values `"a", "b", "c"`, the feedback function will be called 3
        times with kwargs:

        - `{'arg1': 0, arg2: "a"}`,
        - `{'arg1': 1, arg2: "b"}`, 
        - `{'arg1': 2, arg2: "c"}`

    If the quantities of items in the various generators do not match, the
    result will have only as many combinations as the generator with the
    fewest items as per python [zip][zip] (strict mode is not used).

    Note that selectors can use
    [Lens][trulens_eval.utils.serial.Lens] `collect()` to name a single (list)
    value instead of multiple values.
    """

    PRODUCT = "product"
    """Evaluate feedback on all combinations of feedback function arguments.

    Example:
        If the selector for `arg1` generates values `0, 1` and the one for
        `arg2` generates values `"a", "b"`, the feedback function will be called
        4 times with kwargs:

        - `{'arg1': 0, arg2: "a"}`,
        - `{'arg1': 0, arg2: "b"}`,
        - `{'arg1': 1, arg2: "a"}`,
        - `{'arg1': 1, arg2: "b"}`

    See [itertools.product][itertools.product] for more.

    Note that selectors can use
    [Lens][trulens_eval.utils.serial.Lens] `collect()` to name a single (list)
    value instead of multiple values.
    """


class FeedbackDefinition(pyschema.WithClassInfo, serial.SerialModel, Hashable):
    """Serialized parts of a feedback function. 
    
    The non-serialized parts are in the
    [Feedback][trulens_eval.feedback.feedback.Feedback] class.
    """

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    implementation: Optional[Union[pyschema.Function, pyschema.Method]] = None
    """Implementation serialization."""

    aggregator: Optional[Union[pyschema.Function, pyschema.Method]] = None
    """Aggregator method serialization."""

    combinations: Optional[FeedbackCombinations] = FeedbackCombinations.PRODUCT
    """Mode of combining selected values to produce arguments to each feedback
    function call."""

    feedback_definition_id: mod_types_schema.FeedbackDefinitionID
    """Id, if not given, uniquely determined from content."""

    if_exists: Optional[serial.Lens] = None
    """Only execute the feedback function if the following selector names
    something that exists in a record/app.
    
    Can use this to evaluate conditionally on presence of some calls, for
    example. Feedbacks skipped this way will have a status of
    [FeedbackResultStatus.SKIPPED][trulens_eval.schema.feedback.FeedbackResultStatus.SKIPPED].
    """

    if_missing: FeedbackOnMissingParameters = FeedbackOnMissingParameters.ERROR
    """How to handle missing parameters in feedback function calls."""

    selectors: Dict[str, serial.Lens]
    """Selectors; pointers into [Records][trulens_eval.schema.record.Record] of where
    to get arguments for `imp`."""

    supplied_name: Optional[str] = None
    """An optional name. Only will affect displayed tables."""

    higher_is_better: Optional[bool] = None
    """Feedback result magnitude interpretation."""

    def __init__(
        self,
        feedback_definition_id: Optional[mod_types_schema.FeedbackDefinitionID
                                        ] = None,
        implementation: Optional[Union[pyschema.Function,
                                       pyschema.Method]] = None,
        aggregator: Optional[Union[pyschema.Function, pyschema.Method]] = None,
        if_exists: Optional[serial.Lens] = None,
        if_missing: FeedbackOnMissingParameters = FeedbackOnMissingParameters.
        ERROR,
        selectors: Optional[Dict[str, serial.Lens]] = None,
        name: Optional[str] = None,
        higher_is_better: Optional[bool] = None,
        **kwargs
    ):
        selectors = selectors or {}

        if name is not None:
            kwargs['supplied_name'] = name

        super().__init__(
            feedback_definition_id="temporary",
            implementation=implementation,
            aggregator=aggregator,
            selectors=selectors,
            if_exists=if_exists,
            if_missing=if_missing,
            **kwargs
        )

        # By default, higher score is better
        if higher_is_better is None:
            self.higher_is_better = True
        else:
            self.higher_is_better = higher_is_better

        if feedback_definition_id is None:
            if implementation is not None:
                feedback_definition_id = obj_id_of_obj(
                    self.model_dump(), prefix="feedback_definition"
                )
            else:
                feedback_definition_id = "anonymous_feedback_definition"

        self.feedback_definition_id = feedback_definition_id

    def __repr__(self):
        return f"FeedbackDefinition({self.name},\n\tselectors={self.selectors},\n\tif_exists={self.if_exists}\n)"

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(self.feedback_definition_id)

    @property
    def name(self) -> str:
        """Name of the feedback function.
        
        Derived from the name of the serialized implementation function if name
        was not provided.
        """

        if self.supplied_name is not None:
            return self.supplied_name

        if self.implementation is None:
            raise RuntimeError("This feedback function has no implementation.")

        return self.implementation.name


# HACK013: Need these if using __future__.annotations .
FeedbackResult.model_rebuild()
FeedbackCall.model_rebuild()
FeedbackDefinition.model_rebuild()
