from __future__ import annotations

from concurrent.futures import Future
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
from trulens_eval.feedback.provider.base import LLMProvider
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

from trulens_eval.utils.asynchro import sync
from trulens_eval.utils.json import jsonify
from trulens_eval.utils.pyschema import FunctionOrMethod
from trulens_eval.utils.serial import JSON
from trulens_eval.utils.serial import Lens
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_CLOCK
from trulens_eval.utils.text import UNICODE_YIELD
from trulens_eval.utils.threading import TP

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter()


def RAG_triad(
    provider: LLMProvider,
    question: Lens,
    answer: Lens,
    context: Lens
) -> Tuple[Feedback, ...]:
    """
    Creates a triad of feedback functions for evaluating context retrieval generation steps.

    Parameters:
    
    - provider: LLMProvider -- the provider to use for implementing the feedback
      functions,
    
    - question: Lens -- a selector for the question,

    - answer: Lens -- a selector for the answer,

    - context: Lens -- a selector for the context.
    """


    assert hasattr(provider, "relevance"), "Need a provider with the `relevance` feedback function."
    assert hasattr(provider, "qs_relevance"), "Need a provider with the `qs_relevance` feedback function."

    from trulens_eval.feedback.groundedness import Groundedness
    groudedness_provider = Groundedness(groundedness_provider=provider)

    f_groundedness = Feedback(groudedness_provider.groundedness_measure, if_exists=context)\
        .on(context).on(answer)

    f_relevance = Feedback(provider.relevance, if_exists=context)\
        .on(question).on(context)

    f_qa_relevance = Feedback(provider.qs_relevance, if_exists=context)\
        .on(question).on(answer)

    return f_groundedness, f_relevance, f_qa_relevance


# TODO Rename:
# Option:
#  Feedback -> FeedbackRunner
#  FeedbackDefinition -> FeedbackRunnerDefinition
class Feedback(FeedbackDefinition):
    """
    Feedback function container. Typical usage is to specify a feedback
    implementation function from a `Provider` and the mapping of selectors
    describing how to construct the arguments to the implementation:

    ```python
    from trulens_eval import Feedback
    from trulens_eval import Huggingface
    hugs = Huggingface()
    
    # Create a feedback function from a provider:
    feedback = Feedback(
        hugs.language_match # the implementation
    ).on_input_output() # selectors shorthand
    ```

    Attributes include:

        - `imp: Callable` -- an implementation,

        - `agg: Callable` -- an aggregator implementation for handling selectors
          that name more than one value,

        - `higher_is_better: bool` -- whether higher score is better,

        - attributes via parent `FeedbackDefinition`:

            - `feedback_definition_id: str` -- a unique id,

            - `implementation: FunctionOrMethod` -- A serialized version of
              `imp`.

            - `aggregator: FunctionOrMethod` -- A serialized version of `agg`.

            - `supplied_name: str` -- an optional name,

            - `selectors: Dict[str, Lens]` mapping of implementation arguments
              to selectors.
    """

    # Implementation, not serializable, note that FeedbackDefinition contains
    # `implementation` meant to serialize the below.
    imp: Optional[ImpCallable] = pydantic.Field(None, exclude=True)

    # Aggregator method for feedback functions that produce more than one
    # result.
    agg: Optional[AggCallable] = pydantic.Field(None, exclude=True)

    # An optional name. Only will affect display tables
    supplied_name: Optional[str] = None

    # feedback direction
    higher_is_better: Optional[bool] = None

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

                except Exception as e:
                    logger.warning(
                        f"Feedback implementation {imp} cannot be serialized: {e} "
                        f"This may be ok unless you are using the deferred feedback mode."
                    )

                    kwargs['implementation'] = FunctionOrMethod.of_callable(
                        imp, loadable=False
                    )

        else:
            if "implementation" in kwargs:
                imp: ImpCallable = FunctionOrMethod.model_validate(
                    kwargs['implementation']
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
                    # These are for serialization to/from json and for db storage.
                    kwargs['aggregator'] = FunctionOrMethod.of_callable(
                        agg, loadable=False
                    )

        else:
            if kwargs.get('aggregator') is not None:
                agg: AggCallable = FunctionOrMethod.model_validate(
                    kwargs['aggregator']
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

        ret = Feedback.model_copy(self)

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
            record = Record.model_validate(record_json)

            app_json = row.app_json

            if row.get("feedback_json") is None:
                logger.warning(
                    "Cannot evaluate feedback without `feedback_json`. "
                    "This might have come from an old database. \n"
                    f"{row}"
                )
                return None, None

            feedback = Feedback.model_validate(row.feedback_json)

            return feedback, feedback.run_and_log(
                record=record,
                app=app_json,
                tru=tru,
                feedback_result_id=row.feedback_result_id
            )

        feedbacks = db.get_feedback()

        tp = TP()

        futures: List[Future[Tuple[Feedback, FeedbackResult]]] = []

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

        ret = self.model_copy()

        ret.selectors=new_selectors

        return ret

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

        ret = self.model_copy()

        ret.selectors=new_selectors

        return ret

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

        ret = self.model_copy()

        ret.selectors=new_selectors

        return ret

    def run(
        self,
        app: Optional[Union[AppDefinition, JSON]] = None,
        record: Optional[Record] = None,
        source_data: Optional[Dict] = None
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
            record_id=record.record_id if record is not None else "no record",
            name=self.supplied_name
            if self.supplied_name is not None else self.name
        )

        source_data = self._construct_source_data(
            app=app_json, record=record, source_data=source_data
        )

        if self.if_exists is not None:
            if not self.if_exists.exists(source_data):
                logger.warning(f"Feedback {self.name} skipped as {self.if_exists} does not exist.")
                return feedback_result

        # Separate try block for extracting inputs from records/apps in case a
        # user specified something that does not exist. We want to fail and give
        # a warning earlier than later.
        try:
            input_combinations = list(self._extract_selection(source_data=source_data))

        except Exception as e:
            # TODO: Block here to remind us that we may want to do something
            # better here.
            raise e

        try:
            # Total cost, will accumulate.
            cost = Cost()
            multi_result = None

            for ins in input_combinations:
                try:
                    result_and_meta, part_cost = sync(
                        Endpoint.atrack_all_costs_tally,
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

    def _extract_selection(
        self, source_data: Dict
    ) -> Iterable[Dict[str, Any]]:
        
        arg_vals = {}

        for k, q in self.selectors.items():
            try:
                arg_vals[k] = list(q.get(source_data))
            except Exception as e:
                raise RuntimeError(
                    f"Could not locate {q} in recorded data."
                )

        keys = arg_vals.keys()
        vals = arg_vals.values()

        assignments = itertools.product(*vals)

        for assignment in assignments:
            yield {k: v for k, v in zip(keys, assignment)}

        pass

    def _construct_source_data(
        self,
        app: Optional[Union[AppDefinition, JSON]] = None,
        record: Optional[Record] = None,
        source_data: Optional[Dict] = None
    ) -> Dict:
        """
        Combine sources of data to be selected over from three sources: `app`
        structure, `record`, and any extras in a dict `source_data`.
        """
                
        if source_data is None:
            source_data = dict()
        else:
            source_data = dict(source_data) # copy

        if app is not None:
            source_data["__app__"] = app

        if record is not None:
            source_data["__record__"] = record.layout_calls_as_app()

        return source_data
    

    def extract_selection(
        self,
        app: Optional[Union[AppDefinition, JSON]] = None,
        record: Optional[Record] = None,
        source_data: Optional[Dict] = None
    ) -> Iterable[Dict[str, Any]]:
        """
        Given the `app` that produced the given `record`, extract from `record`
        the values that will be sent as arguments to the implementation as
        specified by `self.selectors`. Additional data to select from can be
        provided in `source_data`. All args are optional. If a `Record` is
        specified, its calls are laid out as app (see
        `Record.layout_calls_as_app`).
        """

        return self._extract_selection(
            source_data=self._construct_source_data(
                app=app, record=record, source_data=source_data
            )
        )

        