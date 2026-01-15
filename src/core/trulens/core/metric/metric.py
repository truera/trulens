"""
USER-FACING METRIC CONTAINER: Runtime wrapper that users interact with to create evaluation metrics.
Handles execution, parameter management (examples, criteria, scoring), and database integration.

This is the new primary API for evaluation metrics, replacing the Feedback class.
"""

from __future__ import annotations

from datetime import datetime
import inspect
from inspect import Signature
from inspect import signature
import itertools
import json
import logging
from pprint import pformat
import traceback
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import warnings

import munch
import numpy as np
import pandas
import pydantic
from pydantic import BaseModel
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import pretty_repr
from trulens.core._utils import pycompat as pycompat_utils
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.metric.selector import Selector
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.schema import app as app_schema
from trulens.core.schema import base as base_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import select as select_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import json as json_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import python as python_utils
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils
from trulens.core.utils import threading as threading_utils

if TYPE_CHECKING:
    from trulens.core import session as core_session


logger = logging.getLogger(__name__)
A = TypeVar("A")

ImpCallable = Callable[[A], Union[float, Tuple[float, Dict[str, Any]]]]
"""Signature of metric implementations.

Those take in any number of arguments and return either a single float or a
float and a dictionary (of metadata)."""

AggCallable = Callable[
    [Union[Iterable[float], Iterable[Tuple[float, float]]]], float
]
"""Signature of aggregation functions."""


class SkipEval(Exception):
    """Raised when evaluating a metric function implementation to skip it so
    it is not aggregated with other non-skipped results.

    Args:
        reason: Optional reason for why this evaluation was skipped.

        metric: The Metric instance this run corresponds to.

        ins: The arguments to this run.
    """

    def __init__(
        self,
        reason: Optional[str] = None,
        metric: Optional[Metric] = None,
        ins: Optional[Dict[str, Any]] = None,
    ):
        self.reason = reason
        self.metric = metric
        self.ins = ins

    def __str__(self):
        return "Metric evaluation skipped" + (
            (" because " + self.reason) if self.reason else ""
        )

    def __repr__(self):
        return f"SkipEval(reason={self.reason})"


class InvalidSelector(Exception):
    """Raised when a selector names something that is missing in a record/app."""

    def __init__(
        self,
        selector: serial_utils.Lens,
        source_data: Optional[Dict[str, Any]] = None,
    ):
        self.selector = selector
        self.source_data = source_data

    def __str__(self):
        return f"Selector {self.selector} does not exist in source data."

    def __repr__(self):
        return f"InvalidSelector({self.selector})"


class GroundednessConfigs(BaseModel):
    use_sent_tokenize: bool
    filter_trivial_statements: bool


class Metric(feedback_schema.FeedbackDefinition):
    """Metric function container.

    A metric evaluates some aspect of an AI application's behavior,
    such as relevance, groundedness, or custom quality measures.

    Example:
        ```python
        from trulens.core import Metric, Selector

        # Direct construction with selectors
        metric = Metric(
            name="relevance_v1",
            implementation=my_relevance_fn,
            selectors={
                "query": Selector.select_record_input(),
                "response": Selector.select_record_output(),
            },
        )

        # Or with a provider and fluent API
        from trulens.providers.openai import OpenAI
        provider = OpenAI()

        metric = Metric(
            implementation=provider.relevance
        ).on_input_output()
        ```

    Note:
        The `enable_trace_compression` parameter is only applicable to metrics
        that take 'trace' as an input parameter. It has no effect on other metrics.
    """

    imp: Optional[ImpCallable] = pydantic.Field(None, exclude=True)
    """Implementation callable.

    A serialized version is stored at
    [FeedbackDefinition.implementation][trulens.core.schema.feedback.FeedbackDefinition.implementation].
    """

    agg: Optional[AggCallable] = pydantic.Field(None, exclude=True)
    """Aggregator method for metrics that produce more than one result.

    A serialized version is stored at
    [FeedbackDefinition.aggregator][trulens.core.schema.feedback.FeedbackDefinition.aggregator].
    """

    examples: Optional[List[Tuple]] = pydantic.Field(None, exclude=True)
    """Examples to use when evaluating the metric."""

    criteria: Optional[str] = pydantic.Field(None, exclude=True)
    """Criteria for the metric."""

    additional_instructions: Optional[str] = pydantic.Field(None, exclude=True)
    """Additional instructions for the metric."""

    min_score_val: Optional[int] = pydantic.Field(None, exclude=True)
    """Minimum score value for the metric."""

    max_score_val: Optional[int] = pydantic.Field(None, exclude=True)
    """Maximum score value for the metric."""

    temperature: Optional[float] = pydantic.Field(None, exclude=True)
    """Temperature parameter for the metric."""

    groundedness_configs: Optional[GroundednessConfigs] = pydantic.Field(
        None, exclude=True
    )
    """Optional groundedness configuration parameters."""

    enable_trace_compression: Optional[bool] = pydantic.Field(
        None, exclude=True
    )
    """Whether to compress trace data to reduce token usage when sending traces to metrics.

    When True, traces are compressed to preserve essential information while removing redundant data.
    When False, full uncompressed traces are used. When None (default), the metric's
    default behavior is used. This flag is only applicable to metrics that take
    'trace' as an input parameter."""

    def __init__(
        self,
        implementation: Optional[Callable] = None,
        agg: Optional[Callable] = None,
        examples: Optional[List[Tuple]] = None,
        criteria: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        min_score_val: Optional[int] = 0,
        max_score_val: Optional[int] = 3,
        temperature: Optional[float] = 0.0,
        groundedness_configs: Optional[GroundednessConfigs] = None,
        enable_trace_compression: Optional[bool] = None,
        metric_type: Optional[str] = None,
        description: Optional[str] = None,
        # Legacy parameter name support
        imp: Optional[Callable] = None,
        **kwargs,
    ):
        """Initialize a metric.

        Args:
            implementation: The metric function to execute. Can be a plain
                callable or a method from a Provider class.
            agg: Aggregator function for combining multiple metric results.
                Defaults to np.mean.
            examples: User-supplied examples for this metric.
            criteria: Criteria for the metric evaluation.
            additional_instructions: Custom instructions for the metric.
            min_score_val: Minimum score value (default: 0).
            max_score_val: Maximum score value (default: 3).
            temperature: Temperature parameter for LLM-based metrics (default: 0.0).
            groundedness_configs: Optional groundedness configuration.
            enable_trace_compression: Whether to compress trace data.
            metric_type: Implementation identifier (e.g., "relevance", "groundedness").
                If not provided, defaults to the function name.
            description: Human-readable description of what this metric measures.
            imp: DEPRECATED. Use `implementation` instead.
            **kwargs: Additional arguments passed to parent class.
        """
        # Handle deprecated parameter name
        additional_instructions = deprecation_utils.handle_deprecated_kwarg(
            kwargs,
            "custom_instructions",
            "additional_instructions",
            additional_instructions,
        )

        # Support legacy 'imp' parameter name - if imp is provided, use it as the callable
        if imp is not None:
            # imp is the actual callable to use
            if implementation is None:
                implementation = imp
            elif isinstance(implementation, pyschema_utils.FunctionOrMethod):
                # Both imp (callable) and implementation (serialized) provided
                # Use imp as callable, keep implementation for storage
                kwargs["implementation"] = implementation
                implementation = imp
            elif isinstance(implementation, dict):
                # Both imp and serialized dict provided - use imp, keep dict for storage
                kwargs["implementation"] = implementation
                implementation = imp
            # else: implementation is also a callable, imp takes precedence

        # imp is the python function/method while implementation is a serialized
        # json structure. Create the one that is missing based on the one that
        # is provided:
        if implementation is not None:
            # Check if implementation is already serialized (dict or FunctionOrMethod) or a callable
            if isinstance(implementation, dict):
                # Raw dict - validate and load it
                kwargs["implementation"] = implementation
                implementation = (
                    pyschema_utils.FunctionOrMethod.model_validate(
                        implementation
                    ).load()
                    if implementation is not None
                    else None
                )
            elif isinstance(implementation, pyschema_utils.FunctionOrMethod):
                # Already a FunctionOrMethod object (Pydantic may have converted it)
                # Keep it in kwargs for parent and load the callable
                kwargs["implementation"] = implementation
                implementation = implementation.load()
            else:
                # It's a callable - serialize it for storage
                if "implementation" not in kwargs:
                    try:
                        kwargs["implementation"] = (
                            pyschema_utils.FunctionOrMethod.of_callable(
                                implementation, loadable=True
                            )
                        )

                    except Exception as e:
                        logger.warning(
                            "Metric implementation %s cannot be serialized: %s "
                            "This may be ok unless you are using the deferred feedback mode.",
                            implementation,
                            e,
                        )

                        kwargs["implementation"] = (
                            pyschema_utils.FunctionOrMethod.of_callable(
                                implementation, loadable=False
                            )
                        )

        else:
            if "implementation" in kwargs:
                impl_val = kwargs["implementation"]
                if isinstance(impl_val, pyschema_utils.FunctionOrMethod):
                    # Already a FunctionOrMethod object - just load it
                    implementation = impl_val.load()
                elif impl_val is not None:
                    # Raw dict - validate and load
                    implementation = (
                        pyschema_utils.FunctionOrMethod.model_validate(
                            impl_val
                        ).load()
                    )

        # Similarly with agg and aggregator.
        if agg is not None:
            # Check if agg is already serialized (dict or FunctionOrMethod) or a callable
            if isinstance(agg, dict):
                # Raw dict - validate and load it
                kwargs["aggregator"] = agg
                agg = pyschema_utils.FunctionOrMethod.model_validate(agg).load()
            elif isinstance(agg, pyschema_utils.FunctionOrMethod):
                # Already a FunctionOrMethod object (Pydantic may have converted it)
                kwargs["aggregator"] = agg
                agg = agg.load()
            elif kwargs.get("aggregator") is None:
                try:
                    # These are for serialization to/from json and for db storage.
                    kwargs["aggregator"] = (
                        pyschema_utils.FunctionOrMethod.of_callable(
                            agg, loadable=True
                        )
                    )
                except Exception as e:
                    # User defined functions in script do not have a module so cannot be serialized
                    logger.warning(
                        "Cannot serialize aggregator %s. "
                        "Deferred mode will default to `np.mean` as aggregator. "
                        "If you are not using `FeedbackMode.DEFERRED`, you can safely ignore this warning. "
                        "%s",
                        agg,
                        e,
                    )
                    # These are for serialization to/from json and for db storage.
                    kwargs["aggregator"] = (
                        pyschema_utils.FunctionOrMethod.of_callable(
                            agg, loadable=False
                        )
                    )

        else:
            agg_val = kwargs.get("aggregator")
            if agg_val is not None:
                if isinstance(agg_val, pyschema_utils.FunctionOrMethod):
                    # Already a FunctionOrMethod object - just load it
                    agg = agg_val.load()
                else:
                    # Raw dict - validate and load
                    agg = pyschema_utils.FunctionOrMethod.model_validate(
                        agg_val
                    ).load()
            else:
                # Default aggregator if neither serialized `aggregator` or
                # loaded `agg` were specified.
                agg = np.mean

        # Pass custom parameters to parent class for serialization
        if criteria is not None:
            kwargs["criteria"] = criteria
        if additional_instructions is not None:
            kwargs["additional_instructions"] = additional_instructions
        if examples is not None:
            kwargs["examples"] = examples
        if metric_type is not None:
            kwargs["metric_type"] = metric_type
        if description is not None:
            kwargs["description"] = description

        super().__init__(**kwargs)

        self.imp = implementation
        self.agg = agg
        self.examples = examples
        self.criteria = criteria
        self.additional_instructions = additional_instructions
        self.min_score_val = min_score_val
        self.max_score_val = max_score_val
        self.temperature = temperature
        self.groundedness_configs = groundedness_configs
        self.enable_trace_compression = enable_trace_compression

        # Verify that `imp` expects the arguments specified in `selectors`:
        if self.imp is not None:
            sig: Signature = signature(self.imp)
            for argname in self.selectors.keys():
                assert argname in sig.parameters, (
                    f"{argname} is not an argument to {self.imp.__name__}. "
                    f"Its arguments are {list(sig.parameters.keys())}."
                )

    def on_input_output(self) -> Metric:
        """
        Specifies that the metric implementation arguments are to be the main
        app input and output in that order.

        Returns a new Metric object with the specification.
        """
        return self.on_input().on_output()

    def on_default(self) -> Metric:
        """
        Specifies that one argument metrics should be evaluated on the main
        app output and two argument metrics should be evaluated on main input
        and main output in that order.

        Returns a new Metric object with this specification.
        """

        if self.imp is None:
            raise ValueError(
                "Metric function implementation is required to determine default argument names."
            )
        sig: Signature = signature(self.imp)
        num_remaining_parameters = len(
            list(k for k in sig.parameters.keys() if k not in self.selectors)
        )
        if num_remaining_parameters == 0:
            return self
        if num_remaining_parameters == 1:
            # If there is only one parameter left, we assume it is the output.
            return self.on_output()
        if num_remaining_parameters == 2:
            # If there are two parameters left, we assume they are the input
            # and output.
            return self.on_input_output()
        # If there are more than two parameters left, we cannot guess what to
        # do.
        raise RuntimeError(
            f"Cannot determine default paths for metric function arguments. "
            f"The metric function has signature {sig}."
        )

    def _print_guessed_selector(self, par_name, par_path):
        if is_otel_tracing_enabled():
            return

        if par_path == select_schema.Select.RecordCalls:
            alias_info = " or `Select.RecordCalls`"
        elif par_path == select_schema.Select.RecordInput:
            alias_info = " or `Select.RecordInput`"
        elif par_path == select_schema.Select.RecordOutput:
            alias_info = " or `Select.RecordOutput`"
        else:
            alias_info = ""

        print(
            f"{text_utils.UNICODE_CHECK} In {self.supplied_name if self.supplied_name is not None else self.name}, "
            f"input {par_name} will be set to {par_path}{alias_info} ."
        )

    @staticmethod
    def evaluate_deferred(
        session: core_session.TruSession,
        limit: Optional[int] = None,
        shuffle: bool = False,
        run_location: Optional[feedback_schema.FeedbackRunLocation] = None,
    ) -> List[
        Tuple[
            pandas.Series,
            pycompat_utils.Future[feedback_schema.FeedbackResult],
        ]
    ]:
        """Evaluates metrics that were specified to be deferred.

        Returns a list of tuples with the DB row containing the Metric and
        initial [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] as
        well as the Future which will contain the actual result.

        Args:
            limit: The maximum number of evals to start.

            shuffle: Shuffle the order of the metrics to evaluate.

            run_location: Only run metrics with this run_location.

        Constants that govern behavior:

        - TruSession.RETRY_RUNNING_SECONDS: How long to time before restarting a metric
          that was started but never failed (or failed without recording that
          fact).

        - TruSession.RETRY_FAILED_SECONDS: How long to wait to retry a failed metric.
        """

        db = session.connector.db

        def prepare_metric(
            row,
        ) -> Optional[feedback_schema.FeedbackResultStatus]:
            record_json = row.record_json
            record = record_schema.Record.model_validate(record_json)

            app_json = row.app_json

            if row.get("feedback_json") is None:
                logger.warning(
                    "Cannot evaluate metric without `feedback_json`. "
                    "This might have come from an old database. \n%s",
                    row,
                )
                return None

            metric = Metric.model_validate(row.feedback_json)

            return metric.run_and_log(
                record=record,
                app=app_json,
                session=session,
                feedback_result_id=row.feedback_result_id,
            )

        # Get the different status metrics except those marked DONE.
        metrics_not_done = db.get_feedback(
            status=[
                feedback_schema.FeedbackResultStatus.NONE,
                feedback_schema.FeedbackResultStatus.FAILED,
                feedback_schema.FeedbackResultStatus.RUNNING,
            ],
            limit=limit,
            shuffle=shuffle,
            run_location=run_location,
        )

        tp = threading_utils.TP()

        futures: List[
            Tuple[
                pandas.Series,
                pycompat_utils.Future[feedback_schema.FeedbackResult],
            ]
        ] = []

        for _, row in metrics_not_done.iterrows():
            now = datetime.now().timestamp()
            elapsed = now - row.last_ts

            if row.status == feedback_schema.FeedbackResultStatus.NONE:
                futures.append((row, tp.submit(prepare_metric, row)))

            elif row.status == feedback_schema.FeedbackResultStatus.RUNNING:
                if elapsed > session.RETRY_RUNNING_SECONDS:
                    futures.append((row, tp.submit(prepare_metric, row)))

                else:
                    pass

            elif row.status == feedback_schema.FeedbackResultStatus.FAILED:
                if elapsed > session.RETRY_FAILED_SECONDS:
                    futures.append((row, tp.submit(prepare_metric, row)))

                else:
                    pass

        return futures

    def __call__(self, *args, **kwargs) -> Any:
        assert (
            self.imp is not None
        ), "Metric definition needs an implementation to call."
        if self.examples is not None:
            kwargs["examples"] = self.examples
        if self.criteria is not None:
            kwargs["criteria"] = self.criteria
        if self.additional_instructions is not None:
            kwargs["additional_instructions"] = self.additional_instructions
        if self.min_score_val is not None:
            kwargs["min_score_val"] = self.min_score_val
        if self.max_score_val is not None:
            kwargs["max_score_val"] = self.max_score_val
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.groundedness_configs is not None:
            kwargs["groundedness_configs"] = self.groundedness_configs
        if self.enable_trace_compression is not None:
            kwargs["enable_trace_compression"] = self.enable_trace_compression

        # Filter out unexpected keyword arguments
        sig = signature(self.imp)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return self.imp(*args, **valid_kwargs)

    def aggregate(
        self,
        func: Optional[AggCallable] = None,
        combinations: Optional[feedback_schema.FeedbackCombinations] = None,
    ) -> Metric:
        """
        Specify the aggregation function in case the selectors for this metric
        generate more than one value for implementation argument(s). Can also
        specify the method of producing combinations of values in such cases.

        Returns a new Metric object with the given aggregation function and/or
        the given [combination mode][trulens.core.schema.feedback.FeedbackCombinations].
        """

        if func is None and combinations is None:
            raise ValueError(
                "At least one of `func` or `combinations` must be provided."
            )

        updates = {}
        if func is not None:
            updates["agg"] = func
        if combinations is not None:
            updates["combinations"] = combinations

        return Metric.model_copy(self, update=updates)

    @staticmethod
    def of_feedback_definition(f: feedback_schema.FeedbackDefinition):
        implementation = f.implementation
        aggregator = f.aggregator
        supplied_name = f.supplied_name
        imp_func = implementation.load()
        agg_func = aggregator.load()

        return Metric.model_validate(
            dict(
                imp=imp_func, agg=agg_func, name=supplied_name, **f.model_dump()
            )
        )

    def _next_unselected_arg_name(self):
        if self.imp is not None:
            sig = signature(self.imp)
            par_names = list(
                k for k in sig.parameters.keys() if k not in self.selectors
            )
            if "self" in par_names:
                logger.warning(
                    "Metric function `%s` has `self` as argument. "
                    "Perhaps it is static method or its Provider class was not initialized?",
                    python_utils.callable_name(self.imp),
                )
            if len(par_names) == 0:
                raise TypeError(
                    f"Metric implementation {self.imp} with signature {sig} has no more inputs. "
                    "Perhaps you meant to evaluate it on App output only instead of app input and output?"
                )

            return par_names[0]
        else:
            raise RuntimeError(
                "Cannot determine name of metric function parameter without its definition."
            )

    def on_prompt(self, arg: Optional[str] = None) -> Metric:
        """
        Create a variant of `self` that will take in the main app input or
        "prompt" as input, sending it as an argument `arg` to implementation.
        """
        new_selectors = self.selectors.copy()
        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, select_schema.Select.RecordInput)
        if is_otel_tracing_enabled():
            new_selectors[arg] = Selector.select_record_input()
        else:
            new_selectors[arg] = select_schema.Select.RecordInput
        ret = self.model_copy()
        ret.selectors = new_selectors
        return ret

    # alias
    on_input = on_prompt

    def on_response(self, arg: Optional[str] = None) -> Metric:
        """
        Create a variant of `self` that will take in the main app output or
        "response" as input, sending it as an argument `arg` to implementation.
        """
        new_selectors = self.selectors.copy()
        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, select_schema.Select.RecordOutput)
        if is_otel_tracing_enabled():
            new_selectors[arg] = Selector.select_record_output()
        else:
            new_selectors[arg] = select_schema.Select.RecordOutput
        ret = self.model_copy()
        ret.selectors = new_selectors
        return ret

    # alias
    on_output = on_response

    def on_context(
        self,
        arg: Optional[str] = None,
        *,
        collect_list: bool,
    ):
        """
        Create a variant of `self` that will attempt to take in the context from
        a context retrieval as input, sending it as an argument `arg` to
        implementation.
        """
        if not is_otel_tracing_enabled():
            raise RuntimeError(
                "Context metrics are only supported in OTel mode!"
            )
        new_selectors = self.selectors.copy()
        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, select_schema.Select.RecordOutput)
        new_selectors[arg] = Selector.select_context(collect_list=collect_list)
        ret = self.model_copy()
        ret.selectors = new_selectors
        return ret

    def on(self, *args, **kwargs) -> Metric:
        """
        Create a variant of `self` with the same implementation but the given
        selectors. Those provided positionally get their implementation argument
        name guessed and those provided as kwargs get their name from the kwargs
        key.
        """
        if is_otel_tracing_enabled():
            if len(args) != 1 or len(kwargs) > 0:
                raise ValueError(
                    "OTEL mode only supports a single positional argument to `on`."
                )
            selectors = args[0]
            if not isinstance(selectors, dict):
                raise ValueError(
                    f"OTEL mode only supports dictionary selectors, not {type(selectors)}!"
                )
            sig = signature(self.imp)
            feedback_function_parameters = set(sig.parameters.keys())
            new_selectors = self.selectors.copy()
            for k, v in selectors.items():
                if not isinstance(k, str):
                    raise ValueError(
                        f"OTEL mode only supports string keys, not {type(k)}!"
                    )
                if k not in feedback_function_parameters:
                    raise ValueError(
                        f"Selector key {k} not found in metric function parameters {feedback_function_parameters}!"
                    )
                if not isinstance(v, Selector):
                    raise ValueError(
                        f"OTEL mode only supports Selector values, not {type(v)}!"
                    )
                new_selectors[k] = v
            ret = self.model_copy()
            ret.selectors = new_selectors
            return ret

        new_selectors = self.selectors.copy()

        for k, v in kwargs.items():
            if not isinstance(v, serial_utils.Lens):
                raise ValueError(
                    f"Expected a Lens but got `{v}` of type `{python_utils.class_name(type(v))}`."
                )
            new_selectors[k] = v

        new_selectors.update(kwargs)

        for path in args:
            if not isinstance(path, serial_utils.Lens):
                raise ValueError(
                    f"Expected a Lens but got `{path}` of type `{python_utils.class_name(type(path))}`."
                )

            argname = self._next_unselected_arg_name()
            new_selectors[argname] = path
            self._print_guessed_selector(argname, path)

        ret = self.model_copy()

        ret.selectors = new_selectors

        return ret

    @property
    def sig(self) -> inspect.Signature:
        """Signature of the metric function implementation."""

        if self.imp is None:
            raise RuntimeError(
                "Cannot determine signature of metric function without its definition."
            )

        return signature(self.imp)

    def check_otel_selectors(self):
        if self.imp is None:
            raise RuntimeError(
                "Cannot check selectors of metric function without its definition."
            )

        sig = signature(self.imp)
        function_args = list(sig.parameters.keys())
        required_function_args = [
            param_name
            for param_name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty
            and param.kind
            not in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]
        error_msg = ""
        # Check for extra selectors. Technically, this shouldn't happen ever
        # since we'd fail before this point, but we check it anyway in case
        # things change.
        extra_selectors = []
        for selector in self.selectors:
            if selector not in function_args:
                extra_selectors.append(selector)
        if extra_selectors:
            error_msg += f"Metric function `{self.name}` has selectors that are not in the function signature:\n"
            error_msg += f"Extra selectors: {extra_selectors}\n"
            error_msg += f"Function args: {function_args}\n"
        # Check for missing selectors.
        missing_selectors = []
        for required_function_arg in required_function_args:
            if required_function_arg not in self.selectors:
                missing_selectors.append(required_function_arg)
        if missing_selectors:
            error_msg += (
                f"Metric function `{self.name}` has missing selectors:\n"
            )
            error_msg += f"Missing selectors: {missing_selectors}\n"
            error_msg += f"Required function args: {required_function_args}\n"
        # Throw error if there are any issues.
        if error_msg:
            raise ValueError(error_msg)

    def check_selectors(
        self,
        app: Union[app_schema.AppDefinition, serial_utils.JSON],
        record: record_schema.Record,
        source_data: Optional[Dict[str, Any]] = None,
        warning: bool = False,
    ) -> bool:
        """Check that the selectors are valid for the given app and record.

        Args:
            app: The app that produced the record.

            record: The record that the metric will run on. This can be a
                mostly empty record for checking ahead of producing one. The
                utility method
                [App.dummy_record][trulens.core.app.App.dummy_record] is built
                for this purpose.

            source_data: Additional data to select from when extracting metric
                function arguments.

            warning: Issue a warning instead of raising an error if a selector is
                invalid. As some parts of a Record cannot be known ahead of
                producing it, it may be necessary to not raise exception here
                and only issue a warning.

        Returns:
            True if the selectors are valid. False if not (if warning is set).

        Raises:
            ValueError: If a selector is invalid and warning is not set.
        """

        from trulens.core.app import App

        if source_data is None:
            source_data = {}

        app_type: str = "trulens recorder (`TruChain`, `TruLlama`, etc)"

        if isinstance(app, App):
            app_type = f"`{type(app).__name__}`"
            app = json_utils.jsonify(
                app,
                instrument=app.instrument,
                skip_specials=True,
                redact_keys=True,
            )

        elif isinstance(app, app_schema.AppDefinition):
            app = json_utils.jsonify(app, skip_specials=True, redact_keys=True)

        source_data = self._construct_source_data(
            app=app, record=record, source_data=source_data
        )

        # Build the hint message here.
        msg = ""

        # Keep track whether any selectors failed to validate.
        check_good: bool = True

        for k, q in self.selectors.items():
            if q.exists(source_data):
                continue

            msg += f"""
# Selector check failed

Source of argument `{k}` to `{self.name}` does not exist in app or expected
record:

```python
{q}
# or equivalently
{select_schema.Select.render_for_dashboard(q)}
```

The data used to make this check may be incomplete. If you expect records
produced by your app to contain the selected content, you can ignore this error
by setting `selectors_nocheck` in the {app_type} constructor. Alternatively,
setting `selectors_check_warning` will print out this message but will not raise
an error.

## Additional information:

Metric function signature:
```python
{self.sig}
```

"""
            prefix = q.existing_prefix(source_data)

            if prefix is None:
                continue

            if (
                len(prefix.path) >= 2
                and isinstance(prefix.path[-1], serial_utils.GetItemOrAttribute)
                and prefix.path[-1].get_item_or_attribute() == "rets"
            ):
                # If the selector check failed because the selector was pointing
                # to something beyond the rets of a record call, we have to
                # ignore it as we cannot tell what will be in the rets ahead of
                # invoking app.
                continue

            if (
                len(prefix.path) >= 3
                and isinstance(prefix.path[-2], serial_utils.GetItemOrAttribute)
                and prefix.path[-2].get_item_or_attribute() == "args"
            ):
                # Likewise if failure was because the selector was pointing to
                # method args beyond their parameter names, we also cannot tell
                # their contents so skip.
                continue

            check_good = False

            msg += f"The prefix `{prefix}` selects this data that exists in your app or typical records:\n\n"

            try:
                for prefix_obj in prefix.get(source_data):
                    if isinstance(prefix_obj, munch.Munch):
                        prefix_obj = prefix_obj.toDict()

                    msg += f"- Object of type `{python_utils.class_name(type(prefix_obj))}` starting with:\n"
                    msg += (
                        "```python\n"
                        + text_utils.retab(
                            tab="\t  ",
                            s=pretty_repr(
                                prefix_obj, max_depth=2, indent_size=2
                            ),
                        )
                        + "\n```\n"
                    )

            except Exception as e:  # pylint: disable=W0718
                msg += f"Some non-existent object because: {pretty_repr(e)}"

        if check_good:
            return True

        # Output using rich text.
        rprint(Markdown(msg))

        if warning:
            return False

        else:
            raise ValueError(
                "Some selectors do not exist in the app or record."
            )

    def run(
        self,
        app: Optional[
            Union[app_schema.AppDefinition, serial_utils.JSON]
        ] = None,
        record: Optional[record_schema.Record] = None,
        source_data: Optional[Dict] = None,
        **kwargs: Dict[str, Any],
    ) -> feedback_schema.FeedbackResult:
        """
        Run the metric on the given `record`. The `app` that
        produced the record is also required to determine input/output argument
        names.

        Args:
            app: The app that produced the record. This can be AppDefinition or a jsonized
                AppDefinition. It will be jsonized if it is not already.

            record: The record to evaluate the metric on.

            source_data: Additional data to select from when extracting metric
                function arguments.

            **kwargs: Any additional keyword arguments are used to set or override
                selected metric function inputs.

        Returns:
            A FeedbackResult object with the result of the metric.
        """

        if isinstance(app, app_schema.AppDefinition):
            app_json = json_utils.jsonify(app)
        else:
            app_json = app

        result_vals = []

        feedback_calls = []

        feedback_result = feedback_schema.FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            record_id=record.record_id if record is not None else "no record",
            name=self.supplied_name
            if self.supplied_name is not None
            else self.name,
        )

        source_data = self._construct_source_data(
            app=app_json, record=record, source_data=source_data
        )

        if self.if_exists is not None:
            if not self.if_exists.exists(source_data):
                logger.warning(
                    "Metric %s skipped as %s does not exist.",
                    self.name,
                    self.if_exists,
                )
                feedback_result.status = (
                    feedback_schema.FeedbackResultStatus.SKIPPED
                )
                return feedback_result

        # Separate try block for extracting inputs from records/apps in case a
        # user specified something that does not exist. We want to fail and give
        # a warning earlier than later.
        try:
            input_combinations = list(
                self._extract_selection(
                    source_data=source_data,
                    combinations=self.combinations,
                    **kwargs,
                )
            )

        except InvalidSelector as e:
            # Handle the cases where a selector named something that does not
            # exist in source data.

            if (
                self.if_missing
                == feedback_schema.FeedbackOnMissingParameters.ERROR
            ):
                feedback_result.status = (
                    feedback_schema.FeedbackResultStatus.FAILED
                )
                raise e

            if (
                self.if_missing
                == feedback_schema.FeedbackOnMissingParameters.WARN
            ):
                feedback_result.status = (
                    feedback_schema.FeedbackResultStatus.SKIPPED
                )
                logger.warning(
                    "Metric %s cannot run as %s does not exist in record or app.",
                    self.name,
                    e.selector,
                )
                return feedback_result

            if (
                self.if_missing
                == feedback_schema.FeedbackOnMissingParameters.IGNORE
            ):
                feedback_result.status = (
                    feedback_schema.FeedbackResultStatus.SKIPPED
                )
                return feedback_result

            feedback_result.status = feedback_schema.FeedbackResultStatus.FAILED
            raise ValueError(
                f"Unknown value for `if_missing` {self.if_missing}."
            ) from e

        try:
            # Total cost, will accumulate.
            cost = base_schema.Cost()
            multi_result = None

            # Keep track of evaluations that were skipped due to raising SkipEval.
            skipped_exceptions = []

            for ins in input_combinations:
                try:
                    result_and_meta, get_eval_cost = (
                        core_endpoint.Endpoint.track_all_costs_tally(
                            self, **ins
                        )
                    )

                    cost += get_eval_cost()

                except SkipEval as e:
                    e.metric = self
                    e.ins = ins
                    skipped_exceptions.append(e)
                    warnings.warn(str(e), UserWarning, stacklevel=1)
                    continue  # go to next input_combination

                except Exception as e:
                    raise RuntimeError(
                        f"Evaluation of {self.name} failed on inputs: \n{pformat(ins)[0:128]}."
                    ) from e

                if isinstance(result_and_meta, Tuple):
                    # If output is a tuple of two, we assume it is the float/multifloat and the metadata.
                    assert len(result_and_meta) == 2, (
                        "Metric functions must return either a single float, "
                        "a float-valued dict, or these in combination with a dictionary as a tuple."
                    )
                    result_val, meta = result_and_meta

                    assert isinstance(
                        meta, dict
                    ), f"Metric metadata output must be a dictionary but was {type(meta)}."
                else:
                    # Otherwise it is just the float. We create empty metadata dict.
                    result_val = result_and_meta
                    meta = dict()

                if isinstance(result_val, dict):
                    for val in result_val.values():
                        assert isinstance(val, float), (
                            f"Metric function output with multivalue must be "
                            f"a dict with float values but encountered {type(val)}."
                        )
                    feedback_call = feedback_schema.FeedbackCall(
                        args=ins,
                        ret=np.mean(list(result_val.values())),
                        meta=meta,
                    )

                else:
                    assert isinstance(
                        result_val, (int, float, list, dict)
                    ), f"Metric function output must be a float or an int, a list of floats, or dict but was {type(result_val)}."
                    feedback_call = feedback_schema.FeedbackCall(
                        args=ins, ret=result_val, meta=meta
                    )

                result_vals.append(result_val)
                feedback_calls.append(feedback_call)

            # Warn that there were some skipped evals.
            num_skipped = len(skipped_exceptions)
            num_eval = len(result_vals)
            num_total = num_skipped + num_eval
            if num_skipped > 0 and num_total > 0:
                warnings.warn(
                    (
                        f"{num_skipped}/{num_total}={100.0 * num_skipped / num_total:0.1f}"
                        "% evaluation(s) were skipped because they raised SkipEval "
                        "(see earlier warnings for listing)."
                    ),
                    UserWarning,
                    stacklevel=1,
                )

            if num_eval == 0:
                warnings.warn(
                    f"Metric {self.supplied_name if self.supplied_name is not None else self.name} with aggregation {self.agg} had no inputs.",
                    UserWarning,
                    stacklevel=1,
                )
                result = float("nan")
            else:
                if isinstance(result_vals[0], float):
                    result_vals = np.array(result_vals)
                    result = self.agg(result_vals)
                else:
                    try:
                        # Operates on list of dict; Can be a dict output
                        # (maintain multi) or a float output (convert to single)
                        result = self.agg(result_vals)
                    except Exception:
                        # Alternatively, operate the agg per key
                        result = {}
                        for feedback_output in result_vals:
                            for key in feedback_output:
                                if key not in result:
                                    result[key] = []
                                result[key].append(feedback_output[key])
                        for key in result:
                            result[key] = self.agg(result[key])

                    if isinstance(result, dict):
                        multi_result = result
                        result = np.nan

            feedback_result.update(
                result=result,
                status=feedback_schema.FeedbackResultStatus.DONE,
                cost=cost,
                calls=feedback_calls,
                multi_result=json.dumps(multi_result),
            )

            return feedback_result

        except Exception:
            # Convert traceback to a UTF-8 string, replacing errors to avoid encoding issues
            exc_tb = (
                traceback.format_exc()
                .encode("utf-8", errors="replace")
                .decode("utf-8")
            )
            logger.warning("Metric Function exception caught: %s", exc_tb)
            feedback_result.update(
                error=exc_tb,
                status=feedback_schema.FeedbackResultStatus.FAILED,
            )
            return feedback_result

    def run_and_log(
        self,
        record: record_schema.Record,
        session: core_session.TruSession,
        app: Union[app_schema.AppDefinition, serial_utils.JSON] = None,
        feedback_result_id: Optional[types_schema.FeedbackResultID] = None,
    ) -> Optional[feedback_schema.FeedbackResult]:
        record_id = record.record_id

        db = session.connector.db

        # Placeholder result to indicate a run.
        feedback_result = feedback_schema.FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            feedback_result_id=feedback_result_id,
            record_id=record_id,
            name=self.supplied_name
            if self.supplied_name is not None
            else self.name,
        )

        if feedback_result_id is None:
            feedback_result_id = feedback_result.feedback_result_id

        try:
            db.insert_feedback(
                feedback_result.update(
                    status=feedback_schema.FeedbackResultStatus.RUNNING  # in progress
                )
            )

            feedback_result = self.run(app=app, record=record).update(
                feedback_result_id=feedback_result_id
            )

        except Exception:
            # Convert traceback to a UTF-8 string, replacing errors to avoid encoding issues
            exc_tb = (
                traceback.format_exc()
                .encode("utf-8", errors="replace")
                .decode("utf-8")
            )
            db.insert_feedback(
                feedback_result.update(
                    error=exc_tb,
                    status=feedback_schema.FeedbackResultStatus.FAILED,
                )
            )
            return None

        # Otherwise update based on what Metric.run produced (could be success
        # or failure).
        db.insert_feedback(feedback_result)

        return feedback_result

    @property
    def name(self) -> str:
        """Name of the metric.

        Derived from the name of the function implementing it if no supplied
        name provided.
        """

        if self.supplied_name is not None:
            return self.supplied_name

        if self.imp is not None:
            return self.imp.__name__

        return super().name

    def _extract_selection(
        self,
        source_data: Dict,
        combinations: feedback_schema.FeedbackCombinations = feedback_schema.FeedbackCombinations.PRODUCT,
        **kwargs: Dict[str, Any],
    ) -> Iterable[Dict[str, Any]]:
        """
        Create parameter assignments to self.imp from the given data source or
        optionally additional kwargs.

        Args:
            source_data: The data to select from.

            combinations: How to combine assignments for various variables to
                make an assignment to the whole signature.

            **kwargs: Additional keyword arguments to use instead of looking
                them up from source data. Any parameters specified here will be
                used as the assignment value and the selector for that parameter
                will be ignored.

        """

        arg_vals = {}

        for k, q in self.selectors.items():
            try:
                if k in kwargs:
                    arg_vals[k] = [kwargs[k]]
                else:
                    logger.debug(
                        f"Calling q.get with source_data: {source_data}"
                    )
                    result = q.get(source_data)
                    logger.debug(
                        f"Result of q.get(source_data) for key '{k}': {result}"
                    )
                    arg_vals[k] = list(result)
            except Exception as e:
                raise InvalidSelector(
                    selector=q, source_data=source_data
                ) from e

        # For anything specified in kwargs that did not have a selector, set the
        # assignment here as the above loop will have missed it.
        for k, v in kwargs.items():
            if k not in self.selectors:
                arg_vals[k] = [v]

        keys = arg_vals.keys()
        vals = arg_vals.values()

        if combinations == feedback_schema.FeedbackCombinations.PRODUCT:
            assignments = itertools.product(*vals)
        elif combinations == feedback_schema.FeedbackCombinations.ZIP:
            assignments = zip(*vals)
        else:
            raise ValueError(
                f"Unknown combination mode {combinations}. "
                "Expected `product` or `zip`."
            )

        for assignment in assignments:
            yield {k: v for k, v in zip(keys, assignment)}

    def _construct_source_data(
        self,
        app: Optional[
            Union[app_schema.AppDefinition, serial_utils.JSON]
        ] = None,
        record: Optional[record_schema.Record] = None,
        source_data: Optional[Dict] = None,
        **kwargs: dict,
    ) -> Dict:
        """Combine sources of data to be selected over for metric function inputs.

        Args:
            app: The app that produced the record.

            record: The record to evaluate the metric on.

            source_data: Additional data to select from when extracting metric
                function arguments.

            **kwargs: Any additional keyword arguments are merged into
                source_data.

        Returns:
            A dictionary with the combined data.
        """

        if source_data is None:
            source_data = {}
        else:
            source_data = dict(source_data)  # copy

        source_data.update(kwargs)

        if app is not None:
            source_data["__app__"] = app

        if record is not None:
            source_data["__record__"] = record.layout_calls_as_app()

        return source_data

    def extract_selection(
        self,
        app: Optional[
            Union[app_schema.AppDefinition, serial_utils.JSON]
        ] = None,
        record: Optional[record_schema.Record] = None,
        source_data: Optional[Dict] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Given the `app` that produced the given `record`, extract from `record`
        the values that will be sent as arguments to the implementation as
        specified by `self.selectors`. Additional data to select from can be
        provided in `source_data`. All args are optional. If a
        [Record][trulens.core.schema.record.Record] is specified, its calls are
        laid out as app (see
        [layout_calls_as_app][trulens.core.schema.record.Record.layout_calls_as_app]).
        """

        return self._extract_selection(
            source_data=self._construct_source_data(
                app=app, record=record, source_data=source_data
            )
        )


Metric.model_rebuild()
