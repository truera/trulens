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
    Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
)
import warnings

import munch
import numpy as np
import pandas
import pydantic
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import pretty_repr

from trulens_eval.feedback.provider import base as mod_base_provider
from trulens_eval.feedback.provider.endpoint import base as mod_base_endpoint
from trulens_eval.schema import app as mod_app_schema
from trulens_eval.schema import base as mod_base_schema
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.schema import types as mod_types_schema
from trulens_eval.utils import json as mod_json_utils
from trulens_eval.utils import pyschema as mod_pyschema
from trulens_eval.utils import python as mod_python_utils
from trulens_eval.utils import serial as mod_serial_utils
from trulens_eval.utils import text as mod_text_utils
from trulens_eval.utils import threading as mod_threading_utils

# WARNING: HACK014: importing schema seems to break pydantic for unknown reason.
# This happens even if you import it as something else.
# from trulens_eval import schema # breaks pydantic
# from trulens_eval import schema as tru_schema # also breaks pydantic

logger = logging.getLogger(__name__)

A = TypeVar("A")

ImpCallable = Callable[[A], Union[float, Tuple[float, Dict[str, Any]]]]
"""Signature of feedback implementations.

Those take in any number of arguments and return either a single float or a
float and a dictionary (of metadata)."""

AggCallable = Callable[[Iterable[float]], float]
"""Signature of aggregation functions."""


class InvalidSelector(Exception):
    """Raised when a selector names something that is missing in a record/app."""

    def __init__(
        self,
        selector: mod_serial_utils.Lens,
        source_data: Optional[Dict[str, Any]] = None
    ):
        self.selector = selector
        self.source_data = source_data

    def __str__(self):
        return f"Selector {self.selector} does not exist in source data."

    def __repr__(self):
        return f"InvalidSelector({self.selector})"


def rag_triad(
    provider: mod_base_provider.LLMProvider,
    question: Optional[mod_serial_utils.Lens] = None,
    answer: Optional[mod_serial_utils.Lens] = None,
    context: Optional[mod_serial_utils.Lens] = None
) -> Dict[str, Feedback]:
    """Create a triad of feedback functions for evaluating context retrieval
    generation steps.
    
    If a particular lens is not provided, the relevant selectors will be
    missing. These can be filled in later or the triad can be used for rails
    feedback actions whick fill in the selectors based on specification from
    within colang.

    Args:
        provider: The provider to use for implementing the feedback functions.
    
        question: Selector for the question part.

        answer: Selector for the answer part.

        context: Selector for the context part.
    """

    assert hasattr(
        provider, "relevance"
    ), "Need a provider with the `relevance` feedback function."
    assert hasattr(
        provider, "qs_relevance"
    ), "Need a provider with the `qs_relevance` feedback function."

    are_complete: bool = True

    ret = {}

    for f_imp, f_agg, arg1name, arg1lens, arg2name, arg2lens, f_name in [
        (provider.groundedness_measure_with_cot_reasons, np.mean, "source",
         context.collect(), "statement", answer, "Groundedness"),
        (provider.relevance_with_cot_reasons, np.mean, "prompt", question,
         "response", answer, "Answer Relevance"),
        (provider.context_relevance_with_cot_reasons, np.mean, "question",
         question, "context", context, "Context Relevance")
    ]:
        f = Feedback(f_imp, if_exists=context, name=f_name).aggregate(f_agg)
        if arg1lens is not None:
            f = f.on(**{arg1name: arg1lens})
        else:
            are_complete = False

        if arg2lens is not None:
            f = f.on(**{arg2name: arg2lens})
        else:
            are_complete = False

        ret[f.name] = f

    if not are_complete:
        logger.warning(
            "Some or all RAG triad feedback functions do not have all their selectors set. "
            "This may be ok if they are to be used for colang actions."
        )

    return ret


class Feedback(mod_feedback_schema.FeedbackDefinition):
    """Feedback function container. 
    
    Typical usage is to specify a feedback implementation function from a
    [Provider][trulens_eval.feedback.provider.Provider] and the mapping of
    selectors describing how to construct the arguments to the implementation:

    Example:
        ```python
        from trulens_eval import Feedback
        from trulens_eval import Huggingface
        hugs = Huggingface()
        
        # Create a feedback function from a provider:
        feedback = Feedback(
            hugs.language_match # the implementation
        ).on_input_output() # selectors shorthand
        ```
    """

    imp: Optional[ImpCallable] = pydantic.Field(None, exclude=True)
    """Implementation callable.
    
    A serialized version is stored at
    [FeedbackDefinition.implementation][trulens_eval.schema.feedback.FeedbackDefinition.implementation].
    """

    agg: Optional[AggCallable] = pydantic.Field(None, exclude=True)
    """Aggregator method for feedback functions that produce more than one
    result.
    
    A serialized version is stored at
    [FeedbackDefinition.aggregator][trulens_eval.schema.feedback.FeedbackDefinition.aggregator].
    """

    def __init__(
        self,
        imp: Optional[Callable] = None,
        agg: Optional[Callable] = None,
        **kwargs
    ):

        # imp is the python function/method while implementation is a serialized
        # json structure. Create the one that is missing based on the one that
        # is provided:
        if imp is not None:
            # These are for serialization to/from json and for db storage.
            if 'implementation' not in kwargs:
                try:
                    kwargs['implementation'
                          ] = mod_pyschema.FunctionOrMethod.of_callable(
                              imp, loadable=True
                          )

                except Exception as e:
                    logger.warning(
                        "Feedback implementation %s cannot be serialized: %s "
                        "This may be ok unless you are using the deferred feedback mode.",
                        imp, e
                    )

                    kwargs['implementation'
                          ] = mod_pyschema.FunctionOrMethod.of_callable(
                              imp, loadable=False
                          )

        else:
            if "implementation" in kwargs:
                imp: ImpCallable = mod_pyschema.FunctionOrMethod.model_validate(
                    kwargs['implementation']
                ).load() if kwargs['implementation'] is not None else None

        # Similarly with agg and aggregator.
        if agg is not None:
            if kwargs.get('aggregator') is None:
                try:
                    # These are for serialization to/from json and for db storage.
                    kwargs['aggregator'
                          ] = mod_pyschema.FunctionOrMethod.of_callable(
                              agg, loadable=True
                          )
                except Exception as e:
                    # User defined functions in script do not have a module so cannot be serialized
                    logger.warning(
                        "Cannot serialize aggregator %s. "
                        "Deferred mode will default to `np.mean` as aggregator. "
                        "If you are not using `FeedbackMode.DEFERRED`, you can safely ignore this warning. "
                        "%s", agg, e
                    )
                    # These are for serialization to/from json and for db storage.
                    kwargs['aggregator'
                          ] = mod_pyschema.FunctionOrMethod.of_callable(
                              agg, loadable=False
                          )

        else:
            if kwargs.get('aggregator') is not None:
                agg: AggCallable = mod_pyschema.FunctionOrMethod.model_validate(
                    kwargs['aggregator']
                ).load()
            else:
                # Default aggregator if neither serialized `aggregator` or
                # loaded `agg` were specified.
                agg = np.mean

        super().__init__(**kwargs)

        self.imp = imp
        self.agg = agg

        # Verify that `imp` expects the arguments specified in `selectors`:
        if self.imp is not None:
            sig: Signature = signature(self.imp)
            for argname in self.selectors.keys():
                assert argname in sig.parameters, (
                    f"{argname} is not an argument to {self.imp.__name__}. "
                    f"Its arguments are {list(sig.parameters.keys())}."
                )

    def on_input_output(self) -> Feedback:
        """
        Specifies that the feedback implementation arguments are to be the main
        app input and output in that order.

        Returns a new Feedback object with the specification.
        """
        return self.on_input().on_output()

    def on_default(self) -> Feedback:
        """
        Specifies that one argument feedbacks should be evaluated on the main
        app output and two argument feedbacks should be evaluates on main input
        and main output in that order.

        Returns a new Feedback object with this specification.
        """

        ret = Feedback.model_copy(self)

        ret._default_selectors()

        return ret

    def _print_guessed_selector(self, par_name, par_path):
        if par_path == mod_feedback_schema.Select.RecordCalls:
            alias_info = " or `Select.RecordCalls`"
        elif par_path == mod_feedback_schema.Select.RecordInput:
            alias_info = " or `Select.RecordInput`"
        elif par_path == mod_feedback_schema.Select.RecordOutput:
            alias_info = " or `Select.RecordOutput`"
        else:
            alias_info = ""

        print(
            f"{mod_text_utils.UNICODE_CHECK} In {self.supplied_name if self.supplied_name is not None else self.name}, "
            f"input {par_name} will be set to {par_path}{alias_info} ."
        )

    def _default_selectors(self):
        """
        Fill in default selectors for any remaining feedback function arguments.
        """

        assert self.imp is not None, "Feedback function implementation is required to determine default argument names."

        sig: Signature = signature(self.imp)
        par_names = list(
            k for k in sig.parameters.keys() if k not in self.selectors
        )

        if len(par_names) == 1:
            # A single argument remaining. Assume it is record output.
            selectors = {par_names[0]: mod_feedback_schema.Select.RecordOutput}
            self._print_guessed_selector(
                par_names[0], mod_feedback_schema.Select.RecordOutput
            )

            # TODO: replace with on_output ?

        elif len(par_names) == 2:
            # Two arguments remaining. Assume they are record input and output
            # respectively.
            selectors = {
                par_names[0]: mod_feedback_schema.Select.RecordInput,
                par_names[1]: mod_feedback_schema.Select.RecordOutput
            }
            self._print_guessed_selector(
                par_names[0], mod_feedback_schema.Select.RecordInput
            )
            self._print_guessed_selector(
                par_names[1], mod_feedback_schema.Select.RecordOutput
            )

            # TODO: replace on_input_output ?
        else:
            # Otherwise give up.

            raise RuntimeError(
                f"Cannot determine default paths for feedback function arguments. "
                f"The feedback function has signature {sig}."
            )

        self.selectors = selectors

    @staticmethod
    def evaluate_deferred(
        tru: Tru,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> List[Tuple[pandas.Series, mod_python_utils.Future[mod_feedback_schema.
                                                           FeedbackResult]]]:
        """Evaluates feedback functions that were specified to be deferred.
        
        Returns a list of tuples with the DB row containing the Feedback and
        initial [FeedbackResult][trulens_eval.schema.feedback.FeedbackResult] as
        well as the Future which will contain the actual result.
        
        Args:
            limit: The maximum number of evals to start.

            shuffle: Shuffle the order of the feedbacks to evaluate.
        
        Constants that govern behaviour:

        - Tru.RETRY_RUNNING_SECONDS: How long to time before restarting a feedback
          that was started but never failed (or failed without recording that
          fact).

        - Tru.RETRY_FAILED_SECONDS: How long to wait to retry a failed feedback.
        """

        db = tru.db

        def prepare_feedback(
            row
        ) -> Optional[mod_feedback_schema.FeedbackResultStatus]:
            record_json = row.record_json
            record = mod_record_schema.Record.model_validate(record_json)

            app_json = row.app_json

            if row.get("feedback_json") is None:
                logger.warning(
                    "Cannot evaluate feedback without `feedback_json`. "
                    "This might have come from an old database. \n%s", row
                )
                return None

            feedback = Feedback.model_validate(row.feedback_json)

            return feedback.run_and_log(
                record=record,
                app=app_json,
                tru=tru,
                feedback_result_id=row.feedback_result_id
            )

        # Get the different status feedbacks except those marked DONE.
        feedbacks_not_done = db.get_feedback(
            status=[
                mod_feedback_schema.FeedbackResultStatus.NONE,
                mod_feedback_schema.FeedbackResultStatus.FAILED,
                mod_feedback_schema.FeedbackResultStatus.RUNNING
            ],
            limit=limit,
            shuffle=shuffle,
        )

        tp = mod_threading_utils.TP()

        futures: List[Tuple[
            pandas.Series,
            mod_python_utils.Future[mod_feedback_schema.FeedbackResult]]] = []

        for _, row in feedbacks_not_done.iterrows():
            now = datetime.now().timestamp()
            elapsed = now - row.last_ts

            # TODO: figure out useful things to print.
            # feedback_ident = (
            #     f"[last seen {humanize.naturaldelta(elapsed)} ago] "
            #    f"{row.fname} for app {row.app_json['app_id']}"
            # )

            if row.status == mod_feedback_schema.FeedbackResultStatus.NONE:
                futures.append((row, tp.submit(prepare_feedback, row)))

            elif row.status == mod_feedback_schema.FeedbackResultStatus.RUNNING:

                if elapsed > tru.RETRY_RUNNING_SECONDS:
                    futures.append((row, tp.submit(prepare_feedback, row)))

                else:
                    pass

            elif row.status == mod_feedback_schema.FeedbackResultStatus.FAILED:

                if elapsed > tru.RETRY_FAILED_SECONDS:
                    futures.append((row, tp.submit(prepare_feedback, row)))

                else:
                    pass

        return futures

    def __call__(self, *args, **kwargs) -> Any:
        assert self.imp is not None, "Feedback definition needs an implementation to call."
        return self.imp(*args, **kwargs)

    def aggregate(
        self,
        func: Optional[AggCallable] = None,
        combinations: Optional[mod_feedback_schema.FeedbackCombinations] = None
    ) -> Feedback:
        """
        Specify the aggregation function in case the selectors for this feedback
        generate more than one value for implementation argument(s). Can also
        specify the method of producing combinations of values in such cases.

        Returns a new Feedback object with the given aggregation function and/or
        the given [combination mode][trulens_eval.schema.feedback.FeedbackCombinations].
        """

        if func is None and combinations is None:
            raise ValueError(
                "At least one of `func` or `combinations` must be provided."
            )

        updates = {}
        if func is not None:
            updates['agg'] = func
        if combinations is not None:
            updates['combinations'] = combinations

        return Feedback.model_copy(self, update=updates)

    @staticmethod
    def of_feedback_definition(f: mod_feedback_schema.FeedbackDefinition):
        implementation = f.implementation
        aggregator = f.aggregator
        supplied_name = f.supplied_name
        imp_func = implementation.load()
        agg_func = aggregator.load()

        return Feedback.model_validate(
            dict(
                imp=imp_func,
                agg=agg_func,
                name=supplied_name,
                **f.model_dump()
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
                    "Feedback function `%s` has `self` as argument. "
                    "Perhaps it is static method or its Provider class was not initialized?",
                    mod_python_utils.callable_name(self.imp)
                )
            if len(par_names) == 0:
                raise TypeError(
                    f"Feedback implementation {self.imp} with signature {sig} has no more inputs. "
                    "Perhaps you meant to evalute it on App output only instead of app input and output?"
                )

            return par_names[0]
        else:
            raise RuntimeError(
                "Cannot determine name of feedback function parameter without its definition."
            )

    def on_prompt(self, arg: Optional[str] = None) -> Feedback:
        """
        Create a variant of `self` that will take in the main app input or
        "prompt" as input, sending it as an argument `arg` to implementation.
        """

        new_selectors = self.selectors.copy()

        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(
                arg, mod_feedback_schema.Select.RecordInput
            )

        new_selectors[arg] = mod_feedback_schema.Select.RecordInput

        ret = self.model_copy()

        ret.selectors = new_selectors

        return ret

    # alias
    on_input = on_prompt

    def on_response(self, arg: Optional[str] = None) -> Feedback:
        """
        Create a variant of `self` that will take in the main app output or
        "response" as input, sending it as an argument `arg` to implementation.
        """

        new_selectors = self.selectors.copy()

        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(
                arg, mod_feedback_schema.Select.RecordOutput
            )

        new_selectors[arg] = mod_feedback_schema.Select.RecordOutput

        ret = self.model_copy()

        ret.selectors = new_selectors

        return ret

    # alias
    on_output = on_response

    def on(self, *args, **kwargs) -> Feedback:
        """
        Create a variant of `self` with the same implementation but the given
        selectors. Those provided positionally get their implementation argument
        name guessed and those provided as kwargs get their name from the kwargs
        key.
        """

        new_selectors = self.selectors.copy()

        for k, v in kwargs.items():
            if not isinstance(v, mod_serial_utils.Lens):
                raise ValueError(
                    f"Expected a Lens but got `{v}` of type `{mod_python_utils.class_name(type(v))}`."
                )
            new_selectors[k] = v

        new_selectors.update(kwargs)

        for path in args:
            if not isinstance(path, mod_serial_utils.Lens):
                raise ValueError(
                    f"Expected a Lens but got `{path}` of type `{mod_python_utils.class_name(type(path))}`."
                )

            argname = self._next_unselected_arg_name()
            new_selectors[argname] = path
            self._print_guessed_selector(argname, path)

        ret = self.model_copy()

        ret.selectors = new_selectors

        return ret

    @property
    def sig(self) -> inspect.Signature:
        """Signature of the feedback function implementation."""

        if self.imp is None:
            raise RuntimeError(
                "Cannot determine signature of feedback function without its definition."
            )

        return signature(self.imp)

    def check_selectors(
        self,
        app: Union[mod_app_schema.AppDefinition, mod_serial_utils.JSON],
        record: mod_record_schema.Record,
        source_data: Optional[Dict[str, Any]] = None,
        warning: bool = False
    ) -> bool:
        """Check that the selectors are valid for the given app and record.

        Args:
            app: The app that produced the record.

            record: The record that the feedback will run on. This can be a
                mostly empty record for checking ahead of producing one. The
                utility method
                [App.dummy_record][trulens_eval.app.App.dummy_record] is built
                for this prupose.

            source_data: Additional data to select from when extracting feedback
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

        from trulens_eval.app import App

        if source_data is None:
            source_data = {}

        app_type: str = "trulens recorder (`TruChain`, `TruLlama`, etc)"

        if isinstance(app, App):
            app_type = f"`{type(app).__name__}`"
            app = mod_json_utils.jsonify(
                app,
                instrument=app.instrument,
                skip_specials=True,
                redact_keys=True
            )

        elif isinstance(app, mod_app_schema.AppDefinition):
            app = mod_json_utils.jsonify(
                app, skip_specials=True, redact_keys=True
            )

        source_data = self._construct_source_data(
            app=app, record=record, source_data=source_data
        )

        # Build the hint message here.
        msg = ""

        # Keep track whether any selectors failed to validate.
        check_good: bool = True

        # with c.capture() as cap:
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
{mod_feedback_schema.Select.render_for_dashboard(q)}
```

The data used to make this check may be incomplete. If you expect records
produced by your app to contain the selected content, you can ignore this error
by setting `selectors_nocheck` in the {app_type} constructor. Alternatively,
setting `selectors_check_warning` will print out this message but will not raise
an error.

## Additional information:

Feedback function signature:
```python
{self.sig}
```

"""
            prefix = q.existing_prefix(source_data)

            if prefix is None:
                continue

            if len(prefix.path) >= 2 and isinstance(
                    prefix.path[-1], mod_serial_utils.GetItemOrAttribute
            ) and prefix.path[-1].get_item_or_attribute() == "rets":
                # If the selector check failed because the selector was pointing
                # to something beyond the rets of a record call, we have to
                # ignore it as we cannot tell what will be in the rets ahead of
                # invoking app.
                continue

            if len(prefix.path) >= 3 and isinstance(
                    prefix.path[-2], mod_serial_utils.GetItemOrAttribute
            ) and prefix.path[-2].get_item_or_attribute() == "args":
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

                    msg += f"- Object of type `{mod_python_utils.class_name(type(prefix_obj))}` starting with:\n"
                    msg += "```python\n" + mod_text_utils.retab(
                        tab="\t  ",
                        s=pretty_repr(prefix_obj, max_depth=2, indent_size=2)
                    ) + "\n```\n"

            except Exception as e:
                msg += f"Some non-existant object because: {pretty_repr(e)}"

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
        app: Optional[Union[mod_app_schema.AppDefinition,
                            mod_serial_utils.JSON]] = None,
        record: Optional[mod_record_schema.Record] = None,
        source_data: Optional[Dict] = None,
        **kwargs: Dict[str, Any]
    ) -> mod_feedback_schema.FeedbackResult:
        """
        Run the feedback function on the given `record`. The `app` that
        produced the record is also required to determine input/output argument
        names.

        Args:
            app: The app that produced the record. This can be AppDefinition or a jsonized
                AppDefinition. It will be jsonized if it is not already.

            record: The record to evaluate the feedback on.

            source_data: Additional data to select from when extracting feedback
                function arguments.

            **kwargs: Any additional keyword arguments are used to set or override
                selected feedback function inputs.
            
        Returns:
            A FeedbackResult object with the result of the feedback function.
        """

        if isinstance(app, mod_app_schema.AppDefinition):
            app_json = mod_json_utils.jsonify(app)
        else:
            app_json = app

        result_vals = []

        feedback_calls = []

        feedback_result = mod_feedback_schema.FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            record_id=record.record_id if record is not None else "no record",
            name=self.supplied_name
            if self.supplied_name is not None else self.name
        )

        source_data = self._construct_source_data(
            app=app_json, record=record, source_data=source_data
        )

        if self.if_exists is not None:
            if not self.if_exists.exists(source_data):
                logger.warning(
                    "Feedback %s skipped as %s does not exist.", self.name,
                    self.if_exists
                )
                feedback_result.status = mod_feedback_schema.FeedbackResultStatus.SKIPPED
                return feedback_result

        # Separate try block for extracting inputs from records/apps in case a
        # user specified something that does not exist. We want to fail and give
        # a warning earlier than later.
        try:
            input_combinations = list(
                self._extract_selection(
                    source_data=source_data,
                    combinations=self.combinations,
                    **kwargs
                )
            )

        except InvalidSelector as e:
            # Handle the cases where a selector named something that does not
            # exist in source data.

            if self.if_missing == mod_feedback_schema.FeedbackOnMissingParameters.ERROR:
                feedback_result.status = mod_feedback_schema.FeedbackResultStatus.FAILED
                raise e

            if self.if_missing == mod_feedback_schema.FeedbackOnMissingParameters.WARN:
                feedback_result.status = mod_feedback_schema.FeedbackResultStatus.SKIPPED
                logger.warning(
                    "Feedback %s cannot run as %s does not exist in record or app.",
                    self.name, e.selector
                )
                return feedback_result

            if self.if_missing == mod_feedback_schema.FeedbackOnMissingParameters.IGNORE:
                feedback_result.status = mod_feedback_schema.FeedbackResultStatus.SKIPPED
                return feedback_result

            feedback_result.status = mod_feedback_schema.FeedbackResultStatus.FAILED
            raise ValueError(
                f"Unknown value for `if_missing` {self.if_missing}."
            ) from e

        try:
            # Total cost, will accumulate.
            cost = mod_base_schema.Cost()
            multi_result = None

            for ins in input_combinations:
                try:
                    result_and_meta, part_cost = mod_base_endpoint.Endpoint.track_all_costs_tally(
                        self.imp, **ins
                    )

                    cost += part_cost
                except Exception as e:
                    raise RuntimeError(
                        f"Evaluation of {self.name} failed on inputs: \n{pformat(ins)[0:128]}."
                    ) from e

                if isinstance(result_and_meta, Tuple):
                    # If output is a tuple of two, we assume it is the float/multifloat and the metadata.
                    assert len(result_and_meta) == 2, (
                        "Feedback functions must return either a single float, "
                        "a float-valued dict, or these in combination with a dictionary as a tuple."
                    )
                    result_val, meta = result_and_meta

                    assert isinstance(
                        meta, dict
                    ), f"Feedback metadata output must be a dictionary but was {type(meta)}."
                else:
                    # Otherwise it is just the float. We create empty metadata dict.
                    result_val = result_and_meta
                    meta = dict()

                if isinstance(result_val, dict):
                    for val in result_val.values():
                        assert isinstance(val, float), (
                            f"Feedback function output with multivalue must be "
                            f"a dict with float values but encountered {type(val)}."
                        )
                    feedback_call = mod_feedback_schema.FeedbackCall(
                        args=ins,
                        ret=np.mean(list(result_val.values())),
                        meta=meta
                    )

                else:
                    assert isinstance(
                        result_val, float
                    ), f"Feedback function output must be a float or dict but was {type(result_val)}."
                    feedback_call = mod_feedback_schema.FeedbackCall(
                        args=ins, ret=result_val, meta=meta
                    )

                result_vals.append(result_val)
                feedback_calls.append(feedback_call)

            if len(result_vals) == 0:
                warnings.warn(
                    f"Feedback function {self.supplied_name if self.supplied_name is not None else self.name} with aggregation {self.agg} had no inputs.",
                    UserWarning,
                    stacklevel=1
                )
                result = np.nan

            else:
                if isinstance(result_vals[0], float):
                    result_vals = np.array(result_vals)
                    result = self.agg(result_vals)
                else:
                    try:
                        # Operates on list of dict; Can be a dict output
                        # (maintain multi) or a float output (convert to single)
                        result = self.agg(result_vals)
                    except:
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
                status=mod_feedback_schema.FeedbackResultStatus.DONE,
                cost=cost,
                calls=feedback_calls,
                multi_result=json.dumps(multi_result)
            )

            return feedback_result

        except:
            # Convert traceback to a UTF-8 string, replacing errors to avoid encoding issues
            exc_tb = traceback.format_exc().encode(
                'utf-8', errors='replace'
            ).decode('utf-8')
            logger.warning(f"Feedback Function exception caught: %s", exc_tb)
            feedback_result.update(
                error=exc_tb,
                status=mod_feedback_schema.FeedbackResultStatus.FAILED
            )
            return feedback_result

    def run_and_log(
        self,
        record: mod_record_schema.Record,
        tru: 'Tru',
        app: Union[mod_app_schema.AppDefinition, mod_serial_utils.JSON] = None,
        feedback_result_id: Optional[mod_types_schema.FeedbackResultID] = None
    ) -> Optional[mod_feedback_schema.FeedbackResult]:

        record_id = record.record_id
        app_id = record.app_id

        db = tru.db

        # Placeholder result to indicate a run.
        feedback_result = mod_feedback_schema.FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            feedback_result_id=feedback_result_id,
            record_id=record_id,
            name=self.supplied_name
            if self.supplied_name is not None else self.name
        )

        if feedback_result_id is None:
            feedback_result_id = feedback_result.feedback_result_id

        try:
            db.insert_feedback(
                feedback_result.update(
                    status=mod_feedback_schema.FeedbackResultStatus.
                    RUNNING  # in progress
                )
            )

            feedback_result = self.run(
                app=app, record=record
            ).update(feedback_result_id=feedback_result_id)

        except Exception:
            # Convert traceback to a UTF-8 string, replacing errors to avoid encoding issues
            exc_tb = traceback.format_exc().encode(
                'utf-8', errors='replace'
            ).decode('utf-8')
            db.insert_feedback(
                feedback_result.update(
                    error=exc_tb,
                    status=mod_feedback_schema.FeedbackResultStatus.FAILED
                )
            )
            return

        # Otherwise update based on what Feedback.run produced (could be success
        # or failure).
        db.insert_feedback(feedback_result)

        return feedback_result

    @property
    def name(self) -> str:
        """Name of the feedback function.
        
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
        combinations: mod_feedback_schema.
        FeedbackCombinations = mod_feedback_schema.FeedbackCombinations.PRODUCT,
        **kwargs: Dict[str, Any]
    ) -> Iterable[Dict[str, Any]]:
        """
        Create parameter assignments to self.imp from t he given data source or
        optionally additional kwargs.

        Args:
            source_data: The data to select from.

            combinations: How to combine assignments for various variables to
                make an assignment to the while signature.

            **kwargs: Additional keyword arguments to use instead of looking
                them up from source data. Any parameters specified here will be
                used as the assignment value and the selector for that paremeter
                will be ignored.
        
        """

        arg_vals = {}

        for k, q in self.selectors.items():
            try:
                if k in kwargs:
                    arg_vals[k] = [kwargs[k]]
                else:
                    arg_vals[k] = list(q.get(source_data))
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

        if combinations == mod_feedback_schema.FeedbackCombinations.PRODUCT:
            assignments = itertools.product(*vals)
        elif combinations == mod_feedback_schema.FeedbackCombinations.ZIP:
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
        app: Optional[Union[mod_app_schema.AppDefinition,
                            mod_serial_utils.JSON]] = None,
        record: Optional[mod_record_schema.Record] = None,
        source_data: Optional[Dict] = None,
        **kwargs: dict
    ) -> Dict:
        """Combine sources of data to be selected over for feedback function inputs.

        Args:
            app: The app that produced the record.

            record: The record to evaluate the feedback on.

            source_data: Additional data to select from when extracting feedback
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
        app: Optional[Union[mod_app_schema.AppDefinition,
                            mod_serial_utils.JSON]] = None,
        record: Optional[mod_record_schema.Record] = None,
        source_data: Optional[Dict] = None
    ) -> Iterable[Dict[str, Any]]:
        """
        Given the `app` that produced the given `record`, extract from `record`
        the values that will be sent as arguments to the implementation as
        specified by `self.selectors`. Additional data to select from can be
        provided in `source_data`. All args are optional. If a
        [Record][trulens_eval.schema.record.Record] is specified, its calls are
        laid out as app (see
        [layout_calls_as_app][trulens_eval.schema.record.Record.layout_calls_as_app]).
        """

        return self._extract_selection(
            source_data=self._construct_source_data(
                app=app, record=record, source_data=source_data
            )
        )


Feedback.model_rebuild()
