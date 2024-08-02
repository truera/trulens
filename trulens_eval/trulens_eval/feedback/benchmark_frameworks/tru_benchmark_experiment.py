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

    def context_relevance_to_score(input, output, temperature=0):
        return cortex.context_relevance(question=input, context=output, temperature=temperature)

    benchmark_params = BenchmarkParams(temperature=0.5)
    benchmark_experiment = TruBenchmarkExperiment(ground_truth, context_relevance_to_score, benchmark_params)

    true_labels = benchmark_experiment.load_true_labels()
    mae_agg_func = BenchmarkAggregator(true_labels=true_labels).mae
    """

    def __init__(
        self,
        ground_truth: Union[List, Callable, FunctionOrMethod],
        feedback_to_score_fn: Callable,
        agg_funcs: List[AggCallable],
        benchmark_params: BenchmarkParams,
    ):
        # TODO: instance type check of ground_truth argument + handle groundtruth_impl
        self.ground_truth = ground_truth
        self.feedback_to_score_fn = feedback_to_score_fn
        self.benchmark_params = benchmark_params

        self.f_benchmark_metrics: List[mod_feedback.Feedback] = [
            Feedback(
                lambda x: x,
                name=f"metric_{agg_func.__name__}",
            )
            .on(Select.RecordCalls.run_feedback_on_single_row.rets)
            .aggregate(agg_func)
            for agg_func in agg_funcs
        ]

    def load_beir_dataset(self, *args, **kwargs):
        pass  # TODO

    @instrument
    def run_feedback_on_single_row(
        self, row, feedback_to_score_fn: Callable
    ) -> Union[float, Tuple[float, float]]:
        """Generate a score with the feedback_to_score_fn

        Returns:
            Union[float, Tuple[float, Dict[str, float]]]: feedback score (with metadata) after running the benchmark on a single entry in ground truth data
        """
        benchmark_params = self.benchmark_params.model_dump()
        temperature = benchmark_params.get("temperature", 0)

        ret = feedback_to_score_fn(
            row["query"], row["response"], temperature=temperature
        )

        # TODO: support benchmark parameters beyond temperature in kwargs

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
                    self.run_feedback_on_single_row,
                    row,
                    self.feedback_to_score_fn,
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
