from datetime import datetime
from inspect import Signature
from inspect import signature
import itertools
import json
import logging
import pprint
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pydantic

from trulens_eval.feedback import AggCallable
from trulens_eval.feedback import ImpCallable
from trulens_eval.feedback.provider.endpoint.base import Endpoint
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import Cost
from trulens_eval.schema import FeedbackCall
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import FeedbackResultID
from trulens_eval.schema import FeedbackResultStatus
from trulens_eval.schema import Record
from trulens_eval.schema import Select
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_CLOCK
from trulens_eval.utils.text import UNICODE_YIELD
from trulens_eval.utils.threading import TP

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


# TODO Rename:
# Option:
#  Feedback -> FeedbackRunner
#  FeedbackDefinition -> FeedbackRunnerDefinition
class Feedback(FeedbackDefinition):
    # Implementation, not serializable, note that FeedbackDefinition contains
    # `implementation` meant to serialize the below.
    imp: Optional[ImpCallable] = pydantic.Field(exclude=True)

    # Aggregator method for feedback functions that produce more than one
    # result.
    agg: Optional[AggCallable] = pydantic.Field(exclude=True)

    # An optional name. Only will affect display tables
    supplied_name: Optional[str] = None

    # feedback direction
    higher_is_better: Optional[bool] = pydantic.Field()

    def __init__(
        self,
        imp: Optional[Callable] = None,
        agg: Optional[Callable] = None,
        name: Optional[str] = None,
        higher_is_better: Optional[bool] = None,
        **kwargs
    ):
        """
        A Feedback function container.

        Parameters:
        
        - imp: Optional[Callable] -- implementation of the feedback function.

        - agg: Optional[Callable] -- aggregation function for producing a single
          float for feedback implementations that are run more than once.
        """

        if name is not None:
            kwargs['supplied_name'] = name

        # imp is the python function/method while implementation is a serialized
        # json structure. Create the one that is missing based on the one that
        # is provided:
        if imp is not None:
            # These are for serialization to/from json and for db storage.
            if 'implementation' not in kwargs:
                try:
                    kwargs['implementation'] = FunctionOrMethod.of_callable(
                        imp, loadable=True
                    )
                except ImportError as e:
                    logger.warning(
                        f"Feedback implementation {imp} cannot be serialized: {e}. "
                        f"This may be ok unless you are using the deferred feedback mode."
                    )

                    kwargs['implementation'] = FunctionOrMethod.of_callable(
                        imp, loadable=False
                    )

        else:
            if "implementation" in kwargs:
                imp: ImpCallable = FunctionOrMethod.pick(
                    **(kwargs['implementation'])
                ).load() if kwargs['implementation'] is not None else None

        # Similarly with agg and aggregator.
        if agg is not None:
            if kwargs.get('aggregator') is None:
                try:
                    # These are for serialization to/from json and for db storage.
                    kwargs['aggregator'] = FunctionOrMethod.of_callable(
                        agg, loadable=True
                    )
                except Exception as e:
                    # User defined functions in script do not have a module so cannot be serialized
                    logger.warning(
                        f"Cannot serialize aggregator {agg}. "
                        f"Deferred mode will default to `np.mean` as aggregator. "
                        f"If you are not using FeedbackMode.DEFERRED, you can safely ignore this warning. "
                        f"{e}"
                    )
                    pass
        else:
            if kwargs.get('aggregator') is not None:
                agg: AggCallable = FunctionOrMethod.pick(
                    **(kwargs['aggregator'])
                ).load()
            else:
                # Default aggregator if neither serialized `aggregator` or
                # loaded `agg` were specified.
                agg = np.mean

        super().__init__(**kwargs)

        self.imp = imp
        self.agg = agg

        # By default, higher score is better
        if higher_is_better is None:
            self.higher_is_better = True
        else:
            self.higher_is_better = higher_is_better

        # Verify that `imp` expects the arguments specified in `selectors`:
        if self.imp is not None:
            sig: Signature = signature(self.imp)
            for argname in self.selectors.keys():
                assert argname in sig.parameters, (
                    f"{argname} is not an argument to {self.imp.__name__}. "
                    f"Its arguments are {list(sig.parameters.keys())}."
                )

    def on_input_output(self):
        """
        Specifies that the feedback implementation arguments are to be the main
        app input and output in that order.

        Returns a new Feedback object with the specification.
        """
        return self.on_input().on_output()

    def on_default(self):
        """
        Specifies that one argument feedbacks should be evaluated on the main
        app output and two argument feedbacks should be evaluates on main input
        and main output in that order.

        Returns a new Feedback object with this specification.
        """

        ret = Feedback.parse_obj(self)
        ret._default_selectors()

        return ret

    def _print_guessed_selector(self, par_name, par_path):
        if par_path == Select.RecordCalls:
            alias_info = f" or `Select.RecordCalls`"
        elif par_path == Select.RecordInput:
            alias_info = f" or `Select.RecordInput`"
        elif par_path == Select.RecordOutput:
            alias_info = f" or `Select.RecordOutput`"
        else:
            alias_info = ""

        print(
            f"{UNICODE_CHECK} In {self.supplied_name if self.supplied_name is not None else self.name}, "
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
            selectors = {par_names[0]: Select.RecordOutput}
            self._print_guessed_selector(par_names[0], Select.RecordOutput)

            # TODO: replace with on_output ?

        elif len(par_names) == 2:
            # Two arguments remaining. Assume they are record input and output
            # respectively.
            selectors = {
                par_names[0]: Select.RecordInput,
                par_names[1]: Select.RecordOutput
            }
            self._print_guessed_selector(par_names[0], Select.RecordInput)
            self._print_guessed_selector(par_names[1], Select.RecordOutput)

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
        tru: 'Tru'
    ) -> List['Future[Tuple[Feedback, FeedbackResult]]']:
        """
        Evaluates feedback functions that were specified to be deferred. Returns
        an integer indicating how many evaluates were run.
        """

        db = tru.db

        def prepare_feedback(row):
            record_json = row.record_json
            record = Record(**record_json)

            app_json = row.app_json

            feedback = Feedback(**row.feedback_json)
            return feedback, feedback.run_and_log(
                record=record,
                app=app_json,
                tru=tru,
                feedback_result_id=row.feedback_result_id
            )

        feedbacks = db.get_feedback()

        tp = TP()

        futures: List['Future[Tuple[Feedback, FeedbackResult]]'] = []

        for i, row in feedbacks.iterrows():
            feedback_ident = f"{row.fname} for app {row.app_json['app_id']}, record {row.record_id}"

            if row.status == FeedbackResultStatus.NONE:

                print(
                    f"{UNICODE_YIELD} Feedback task starting: {feedback_ident}"
                )

                futures.append(tp.submit(prepare_feedback, row))

            elif row.status in [FeedbackResultStatus.RUNNING]:
                now = datetime.now().timestamp()
                if now - row.last_ts > 30:
                    print(
                        f"{UNICODE_YIELD} Feedback task last made progress over 30 seconds ago. "
                        f"Retrying: {feedback_ident}"
                    )
                    futures.append(tp.submit(prepare_feedback, row))

                else:
                    print(
                        f"{UNICODE_CLOCK} Feedback task last made progress less than 30 seconds ago. "
                        f"Giving it more time: {feedback_ident}"
                    )

            elif row.status in [FeedbackResultStatus.FAILED]:
                now = datetime.now().timestamp()
                if now - row.last_ts > 60 * 5:
                    print(
                        f"{UNICODE_YIELD} Feedback task last made progress over 5 minutes ago. "
                        f"Retrying: {feedback_ident}"
                    )
                    futures.append(tp.submit(prepare_feedback, row))

                else:
                    print(
                        f"{UNICODE_CLOCK} Feedback task last made progress less than 5 minutes ago. "
                        f"Not touching it for now: {feedback_ident}"
                    )

            elif row.status == FeedbackResultStatus.DONE:
                pass

        return futures

    def __call__(self, *args, **kwargs) -> Any:
        assert self.imp is not None, "Feedback definition needs an implementation to call."
        return self.imp(*args, **kwargs)

    def aggregate(self, func: Callable) -> 'Feedback':
        """
        Specify the aggregation function in case the selectors for this feedback
        generate more than one value for implementation argument(s).

        Returns a new Feedback object with the given aggregation function.
        """

        return Feedback(
            imp=self.imp,
            selectors=self.selectors,
            agg=func,
            name=self.supplied_name,
            higher_is_better=self.higher_is_better
        )

    @staticmethod
    def of_feedback_definition(f: FeedbackDefinition):
        implementation = f.implementation
        aggregator = f.aggregator
        supplied_name = f.supplied_name
        imp_func = implementation.load()
        agg_func = aggregator.load()

        return Feedback(
            imp=imp_func, agg=agg_func, name=supplied_name, **f.dict()
        )

    def _next_unselected_arg_name(self):
        if self.imp is not None:
            sig = signature(self.imp)
            par_names = list(
                k for k in sig.parameters.keys() if k not in self.selectors
            )
            if "self" in par_names:
                logger.warning(
                    f"Feedback function `{self.imp.__name__}` has `self` as argument. "
                    "Perhaps it is static method or its Provider class was not initialized?"
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

    def on_prompt(self, arg: Optional[str] = None):
        """
        Create a variant of `self` that will take in the main app input or
        "prompt" as input, sending it as an argument `arg` to implementation.
        """

        new_selectors = self.selectors.copy()

        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, Select.RecordInput)

        new_selectors[arg] = Select.RecordInput

        return Feedback(
            imp=self.imp,
            selectors=new_selectors,
            agg=self.agg,
            name=self.supplied_name,
            higher_is_better=self.higher_is_better
        )

    on_input = on_prompt

    def on_response(self, arg: Optional[str] = None):
        """
        Create a variant of `self` that will take in the main app output or
        "response" as input, sending it as an argument `arg` to implementation.
        """

        new_selectors = self.selectors.copy()

        if arg is None:
            arg = self._next_unselected_arg_name()
            self._print_guessed_selector(arg, Select.RecordOutput)

        new_selectors[arg] = Select.RecordOutput

        return Feedback(
            imp=self.imp,
            selectors=new_selectors,
            agg=self.agg,
            name=self.supplied_name,
            higher_is_better=self.higher_is_better
        )

    on_output = on_response

    def on(self, *args, **kwargs):
        """
        Create a variant of `self` with the same implementation but the given
        selectors. Those provided positionally get their implementation argument
        name guessed and those provided as kwargs get their name from the kwargs
        key.
        """

        new_selectors = self.selectors.copy()
        new_selectors.update(kwargs)

        for path in args:
            argname = self._next_unselected_arg_name()
            new_selectors[argname] = path
            self._print_guessed_selector(argname, path)

        return Feedback(
            imp=self.imp,
            selectors=new_selectors,
            agg=self.agg,
            name=self.supplied_name,
            higher_is_better=self.higher_is_better
        )

    def run(
        self, app: Union[AppDefinition, JSON], record: Record
    ) -> FeedbackResult:
        """
        Run the feedback function on the given `record`. The `app` that
        produced the record is also required to determine input/output argument
        names.

        Might not have a AppDefinitionhere but only the serialized app_json .
        """

        if isinstance(app, AppDefinition):
            app_json = jsonify(app)
        else:
            app_json = app

        result_vals = []

        feedback_calls = []

        feedback_result = FeedbackResult(
            feedback_definition_id=self.feedback_definition_id,
            record_id=record.record_id,
            name=self.supplied_name
            if self.supplied_name is not None else self.name
        )

        # Separate try block for extracting inputs from records/apps in case a
        # user specified something that does not exist. We want to fail and give
        # a warning earlier than later.
        try:
            input_combinations = list(
                self.extract_selection(app=app_json, record=record)
            )
        except Exception as e:
            print(e)
            raise e

        try:
            # Total cost, will accumulate.
            cost = Cost()
            multi_result = None

            for ins in input_combinations:
                try:
                    result_and_meta, part_cost = Endpoint.track_all_costs_tally(
                        lambda: self.imp(**ins)
                    )
                    cost += part_cost
                except Exception as e:
                    raise RuntimeError(
                        f"Evaluation of {self.name} failed on inputs: \n{pp.pformat(ins)[0:128]}\n{e}."
                    )

                if isinstance(result_and_meta, Tuple):
                    # If output is a tuple of two, we assume it is the float/multifloat and the metadata.
                    assert len(result_and_meta) == 2, (
                        f"Feedback functions must return either a single float, "
                        f"a float-valued dict, or these in combination with a dictionary as a tuple."
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
                    feedback_call = FeedbackCall(
                        args=ins,
                        ret=np.mean(list(result_val.values())),
                        meta=meta
                    )

                else:
                    assert isinstance(
                        result_val, float
                    ), f"Feedback function output must be a float or dict but was {type(result_val)}."
                    feedback_call = FeedbackCall(
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
                status=FeedbackResultStatus.DONE,
                cost=cost,
                calls=feedback_calls,
                multi_result=json.dumps(multi_result)
            )

            return feedback_result

        except:
            exc_tb = traceback.format_exc()
            logger.warning(f"Feedback Function exception caught: {exc_tb}")
            feedback_result.update(
                error=exc_tb, status=FeedbackResultStatus.FAILED
            )
            return feedback_result

    def run_and_log(
        self,
        record: Record,
        tru: 'Tru',
        app: Union[AppDefinition, JSON] = None,
        feedback_result_id: Optional[FeedbackResultID] = None
    ) -> Optional[FeedbackResult]:
        record_id = record.record_id
        app_id = record.app_id

        db = tru.db

        # Placeholder result to indicate a run.
        feedback_result = FeedbackResult(
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
                    status=FeedbackResultStatus.RUNNING  # in progress
                )
            )

            feedback_result = self.run(
                app=app, record=record
            ).update(feedback_result_id=feedback_result_id)

        except Exception as e:
            exc_tb = traceback.format_exc()
            db.insert_feedback(
                feedback_result.update(
                    error=exc_tb, status=FeedbackResultStatus.FAILED
                )
            )
            return

        # Otherwise update based on what Feedback.run produced (could be success or failure).
        db.insert_feedback(feedback_result)

        return feedback_result

    @property
    def name(self):
        """
        Name of the feedback function. Presently derived from the name of the
        function implementing it.
        """

        if self.imp is None:
            raise RuntimeError("This feedback function has no implementation.")

        return self.imp.__name__

    def extract_selection(
        self, app: Union[AppDefinition, JSON], record: Record
    ) -> Iterable[Dict[str, Any]]:
        """
        Given the `app` that produced the given `record`, extract from
        `record` the values that will be sent as arguments to the implementation
        as specified by `self.selectors`.
        """

        arg_vals = {}

        for k, v in self.selectors.items():
            if isinstance(v, Select.Query):
                q = v

            else:
                raise RuntimeError(f"Unhandled selection type {type(v)}.")

            if q.path[0] == Select.Record.path[0]:
                o = record.layout_calls_as_app()
            elif q.path[0] == Select.App.path[0]:
                o = app
            else:
                raise ValueError(
                    f"Query {q} does not indicate whether it is about a record or about a app."
                )

            q_within_o = Select.Query(path=q.path[1:])
            try:
                arg_vals[k] = list(q_within_o.get(o))
            except Exception as e:
                raise RuntimeError(
                    f"Could not locate {q_within_o} in app/record."
                )

        keys = arg_vals.keys()
        vals = arg_vals.values()

        assignments = itertools.product(*vals)

        for assignment in assignments:
            yield {k: v for k, v in zip(keys, assignment)}
