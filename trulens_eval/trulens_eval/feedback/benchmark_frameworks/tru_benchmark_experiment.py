from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from trulens_eval import feedback as mod_feedback
from trulens_eval import Feedback
from trulens_eval import Select
from trulens_eval.feedback import Feedback
from trulens_eval.feedback.feedback import AggCallable
from trulens_eval.tru_custom_app import instrument
from trulens_eval.utils.pyschema import FunctionOrMethod

log = logging.getLogger(__name__)


class BenchmarkParams(BaseModel):
    temperature: float = 0.0
    criteria: Optional[str] = None
    output_space: Optional[str] = None
    # TODO: support more parameters
    # "use_verb_confidence": False,
    # K should not be part of benchmark params b/c each set of benchmark params could have multiple set of K values for different metric aggregators


class TruBenchmarkExperiment:
    """
    Example usage:

    cortex = Cortex(model_engine="snowflake-arctic")

    def context_relevance_ff_to_score(input, output, temperature=0):
        return cortex.context_relevance(question=input, context=output, temperature=temperature)


    tru_labels = [1, 0, 0, ...] # ground truth labels
    mae_agg_func = GroundTruthAggregator(true_labels=true_labels).mae

    tru_benchmark_artic = tru.BenchmarkExperiment(
        app_id="MAE",
        ground_truth=golden_set,
        feedback_fn=context_relevance_ff_to_score,
        agg_funcs=[mae_agg_func],
        benchmark_params=BenchmarkParams(temperature=0.5),
    )
    """

    def __init__(
        self,
        ground_truth: Union[List, Callable, FunctionOrMethod],
        feedback_fn: Callable,
        agg_funcs: List[AggCallable],
        benchmark_params: BenchmarkParams,
    ):
        """Create a benchmark experiment class which defines custom
        feedback functions and aggregators to evaluate the feedback function on a ground truth dataset.
        Args:
            ground_truth (Union[List, Callable, FunctionOrMethod]): ground truth data to evaluate the feedback function on
            feedback_fn (Callable): function that takes in a row of ground truth data and returns a score by typically a LLM-as-judge
            agg_funcs (List[AggCallable]): list of aggregation functions to compute metrics on the feedback scores
            benchmark_params (BenchmarkParams): benchmark configuration parameters
        """
        # TODO: instance type check of ground_truth argument + handle groundtruth_impl
        self.ground_truth = ground_truth
        self.feedback_fn = feedback_fn
        self.benchmark_params = benchmark_params

        self.f_benchmark_metrics: List[mod_feedback.Feedback] = [
            Feedback(
                lambda x: x,
                name=f"metric_{agg_func.__name__}",
            )
            .on(Select.RecordCalls.run_score_generation_on_single_row.rets)
            .aggregate(agg_func)
            for agg_func in agg_funcs
        ]

    def load_beir_dataset(self, *args, **kwargs):
        pass  # TODO

    @instrument
    def run_score_generation_on_single_row(
        self, row, feedback_fn: Callable
    ) -> Union[float, Tuple[float, float]]:
        """Generate a score with the feedback_fn

        Returns:
            Union[float, Tuple[float, Dict[str, float]]]: feedback score (with metadata) after running the benchmark on a single entry in ground truth data
        """

        benchmark_params_dict: dict = self.benchmark_params.model_dump()
        ret = feedback_fn(row["query"], row["response"], benchmark_params_dict)

        # TODO: better define the shape of arguments of feedback_fn

        if not isinstance(ret, tuple) and not isinstance(ret, float):
            raise ValueError(
                f"Output must be a float or a tuple, got {type(ret)}"
            )

        if isinstance(ret, tuple) and isinstance(ret[1], dict):
            ret = (
                ret[0],
                list(ret[1].values())[-1],
            )  # this is the case when a feedback function returns a tuple with a score and metadata like (0.5, {"confidence_score": 0.8})
        return ret

    @instrument
    def collect_feedback_scores(
        self,
    ) -> Union[
        List[float], List[Tuple[float]], Tuple[List[float], List[float]]
    ]:
        """Collect the list of generated feedback scores as input to the benchmark aggregation functions

        Returns:
            List[float]: feedback scores after running the benchmark on all entries in ground truth data
        """
        scores = []
        meta_scores = []
        with ThreadPoolExecutor() as executor:
            future_to_row = {
                executor.submit(
                    self.run_score_generation_on_single_row,
                    row,
                    self.feedback_fn,
                ): row
                for row in self.ground_truth
            }
            for future in as_completed(future_to_row):
                try:
                    ret = future.result()

                    if isinstance(ret, float):
                        score = ret
                    else:
                        score, metadata = ret

                        meta_scores.append(metadata)

                    scores.append(score)

                except Exception as e:
                    log.error(f"Row generated an exception: {e}")

        if meta_scores:
            return scores, meta_scores
        else:
            return scores
