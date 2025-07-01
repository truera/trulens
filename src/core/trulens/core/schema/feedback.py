"""Serializable feedback-related classes."""

from __future__ import annotations

import datetime
from enum import Enum
import logging
from pprint import pformat
from typing import (
    Any,
    ClassVar,
    Dict,
    Hashable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import pydantic
from trulens.core.schema import base as base_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils

T = TypeVar("T")

logger = logging.getLogger(__name__)


class FeedbackMode(str, Enum):
    """Mode of feedback evaluation.

    Specify this using the `feedback_mode` to [App][trulens.core.app.App]
    constructors.

    !!! Note
        This class extends [str][str] to allow users to compare its values with
        their string representations, i.e. in `if mode == "none": ...`. Internal
        uses should use the enum instances.
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
    `TruSession.start_deferred_feedback_evaluator`."""


class FeedbackRunLocation(str, Enum):
    """Where the feedback evaluation takes place (e.g. locally, at a Snowflake server, etc)."""

    IN_APP = "in_app"
    """Run on the same process (or child process) of the app invocation."""

    SNOWFLAKE = "snowflake"
    """Run on a Snowflake server."""


class FeedbackResultStatus(str, Enum):
    """For deferred feedback evaluation, these values indicate status of
    evaluation.

    !!! Note
        This class extends [str][str] to allow users to compare its values with
        their string representations, i.e. in `if status == "done": ...`. Internal
        uses should use the enum instances.
    """

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

    !!! Note
        This class extends [str][str] to allow users to compare its values with
        their string representations, i.e. in `if onmissing == "error": ...`.
        Internal uses should use the enum instances.
    """

    ERROR = "error"
    """Raise an error if a parameter is missing.

    The result status will be set to
    [FAILED][trulens.core.schema.feedback.FeedbackResultStatus.FAILED].
    """

    WARN = "warn"
    """Warn if a parameter is missing.

    The result status will be set to
    [SKIPPED][trulens.core.schema.feedback.FeedbackResultStatus.SKIPPED].
    """

    IGNORE = "ignore"
    """Do nothing.

    No warning or error message will be shown. The result status will be set to
    [SKIPPED][trulens.core.schema.feedback.FeedbackResultStatus.SKIPPED].
    """


class FeedbackCall(serial_utils.SerialModel):
    """Invocations of feedback function results in one of these instances.

    Note that a single `Feedback` instance might require more than one call.
    """

    args: Dict[str, Optional[serial_utils.JSON]] = pydantic.Field(
        default_factory=dict
    )
    """Arguments to the feedback function."""

    ret: Union[float, List[float], List[Tuple], List[Any]] = pydantic.Field(
        default=0.0
    )
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
            out += f"{tab}meta = \n{text_utils.retab(tab=tab * 2, s=pformat(self.meta))}\n"

        return out

    def __repr__(self) -> str:
        return str(self)


class FeedbackResult(serial_utils.SerialModel):
    """Feedback results for a single [Feedback][trulens.core.Feedback] instance.

    This might involve multiple feedback function calls. Typically you should
    not be constructing these objects yourself except for the cases where you'd
    like to log human feedback.

    Attributes:
        feedback_result_id: Unique identifier for this result.

        record_id: Record over which the feedback was evaluated.

        feedback_definition_id: The id of the
            [FeedbackDefinition][trulens.core.schema.feedback.FeedbackDefinition] which
            was evaluated to get this result.

        last_ts: Last timestamp involved in the evaluation.

        status: For deferred feedback evaluation, the
            status of the evaluation.

        cost: Cost of the evaluation.

        name: Given name of the feedback.

        calls: Individual feedback function invocations.

        result: Final result, potentially aggregating multiple calls.

        error: Error information if there was an error.

        multi_result: TBD
    """

    feedback_result_id: types_schema.FeedbackResultID

    # Record over which the feedback was evaluated.
    record_id: types_schema.RecordID

    # The `Feedback` / `FeedbackDefinition` which was evaluated to get this
    # result.
    feedback_definition_id: Optional[types_schema.FeedbackDefinitionID] = None

    # Last timestamp involved in the evaluation.
    last_ts: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now
    )

    status: FeedbackResultStatus = FeedbackResultStatus.NONE
    """For deferred feedback evaluation, the status of the evaluation."""

    cost: base_schema.Cost = pydantic.Field(default_factory=base_schema.Cost)

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
        feedback_result_id: Optional[types_schema.FeedbackResultID] = None,
        **kwargs,
    ):
        super().__init__(feedback_result_id="temporary", **kwargs)

        if feedback_result_id is None:
            feedback_result_id = json_utils.obj_id_of_obj(
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
    [FeedbackDefinition][trulens.core.schema.feedback.FeedbackDefinition] and can be
    specified with
    [Feedback.aggregate][trulens.core.Feedback.aggregate].
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
    fewest items as per Python [zip][zip] (strict mode is not used).

    Note that selectors can use
    [Lens][trulens.core.utils.serial.Lens] `collect()` to name a single (list)
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
    [Lens][trulens.core.utils.serial.Lens] `collect()` to name a single (list)
    value instead of multiple values.
    """


class FeedbackDefinition(
    pyschema_utils.WithClassInfo, serial_utils.SerialModel, Hashable
):
    """Serialized parts of a feedback function.

    The non-serialized parts are in the
    [Feedback][trulens.core.Feedback] class.
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

    implementation: Optional[
        Union[pyschema_utils.Function, pyschema_utils.Method]
    ] = None
    """Implementation serialization."""

    aggregator: Optional[
        Union[pyschema_utils.Function, pyschema_utils.Method]
    ] = None
    """Aggregator method serialization."""

    examples: Optional[List[Tuple]] = None
    """User supplied examples for this feedback function."""

    criteria: Optional[str] = None
    """Criteria for the feedback function."""

    combinations: Optional[FeedbackCombinations] = FeedbackCombinations.PRODUCT
    """Mode of combining selected values to produce arguments to each feedback
    function call."""

    feedback_definition_id: types_schema.FeedbackDefinitionID
    """Id, if not given, uniquely determined from content."""

    if_exists: Optional[serial_utils.Lens] = None
    """Only execute the feedback function if the following selector names
    something that exists in a record/app.

    Can use this to evaluate conditionally on presence of some calls, for
    example. Feedbacks skipped this way will have a status of
    [FeedbackResultStatus.SKIPPED][trulens.core.schema.feedback.FeedbackResultStatus.SKIPPED].
    """

    if_missing: FeedbackOnMissingParameters = FeedbackOnMissingParameters.ERROR
    """How to handle missing parameters in feedback function calls."""

    run_location: Optional[FeedbackRunLocation]
    """Where the feedback evaluation takes place (e.g. locally, at a Snowflake server, etc)."""

    selectors: Dict[str, serial_utils.Lens]
    """Selectors; pointers into [Records][trulens.core.schema.record.Record] of where
    to get arguments for `imp`."""

    supplied_name: Optional[str] = None
    """An optional name. Only will affect displayed tables."""

    higher_is_better: Optional[bool] = None
    """Feedback result magnitude interpretation."""

    def __init__(
        self,
        feedback_definition_id: Optional[
            types_schema.FeedbackDefinitionID
        ] = None,
        implementation: Optional[
            Union[pyschema_utils.Function, pyschema_utils.Method]
        ] = None,
        aggregator: Optional[
            Union[pyschema_utils.Function, pyschema_utils.Method]
        ] = None,
        examples: Optional[List[Tuple]] = None,
        criteria: Optional[str] = None,
        if_exists: Optional[serial_utils.Lens] = None,
        if_missing: FeedbackOnMissingParameters = FeedbackOnMissingParameters.ERROR,
        selectors: Optional[Dict[str, serial_utils.Lens]] = None,
        name: Optional[str] = None,
        higher_is_better: Optional[bool] = None,
        run_location: Optional[FeedbackRunLocation] = None,
        **kwargs,
    ):
        selectors = selectors or {}

        if name is not None:
            kwargs["supplied_name"] = name

        super().__init__(
            feedback_definition_id="temporary",
            implementation=implementation,
            aggregator=aggregator,
            examples=examples,
            criteria=criteria,
            selectors=selectors,
            if_exists=if_exists,
            if_missing=if_missing,
            run_location=run_location,
            **kwargs,
        )

        # By default, higher score is better
        if higher_is_better is None:
            self.higher_is_better = True
        else:
            self.higher_is_better = higher_is_better

        if feedback_definition_id is None:
            if implementation is not None:
                feedback_definition_id = json_utils.obj_id_of_obj(
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
