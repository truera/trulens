from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel
from trulens.apps.custom import TruCustomApp
from trulens.apps.custom import instrument
from trulens.core import Feedback
from trulens.core import Select
from trulens.core.feedback.feedback import AggCallable

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
    !!! example
        ``` python
        snowflake_connection_parameters = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ["SNOWFLAKE_DATABASE"],
            "schema": os.environ["SNOWFLAKE_SCHEMA"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        }
        cortex = Cortex(
            snowflake.connector.connect(**snowflake_connection_parameters)
            model_engine="snowflake-arctic",
        )

        def context_relevance_ff_to_score(input, output, temperature=0):
            return cortex.context_relevance(question=input, context=output, temperature=temperature)

        tru_labels = [1, 0, 0, ...] # ground truth labels collected from ground truth data collection
        mae_agg_func = GroundTruthAggregator(true_labels=true_labels).mae

        tru_benchmark_arctic = session.BenchmarkExperiment(
            app_name="MAE",
            feedback_fn=context_relevance_ff_to_score,
            agg_funcs=[mae_agg_func],
            benchmark_params=BenchmarkParams(temperature=0.5),
        )
        ```
    """

    def __init__(
        self,
        feedback_fn: Callable,
        agg_funcs: List[AggCallable],
        benchmark_params: BenchmarkParams,
    ):
        """Create a benchmark experiment class which defines custom
        feedback functions and aggregators to evaluate the feedback function on a ground truth dataset.

        Args:
            feedback_fn (Callable): function that takes in a row of ground truth data and returns a score by typically a LLM-as-judge
            agg_funcs (List[AggCallable]): list of aggregation functions to compute metrics on the feedback scores
            benchmark_params (BenchmarkParams): benchmark configuration parameters

        """

        self.feedback_fn = feedback_fn
        self.benchmark_params = benchmark_params

        self.f_benchmark_metrics: List[Feedback] = [
            Feedback(
                lambda x: x,
                name=f"metric_{agg_func.__name__}",
            )
            .on(Select.RecordCalls.run_score_generation_on_single_row.rets)
            .aggregate(agg_func)
            for agg_func in agg_funcs
        ]

    @instrument
    def run_score_generation_on_single_row(
        self,
        feedback_fn: Callable,
        feedback_args: List[Any],
    ) -> Union[float, Tuple[float, float]]:
        """Generate a score with the feedback_fn

        Args:
            row: A single row from the dataset.
            feedback_fn: The function used to generate feedback scores.

        Returns:
            Union[float, Tuple[float, float]]: Feedback score (with metadata) after running the benchmark on a single entry in ground truth data.
        """

        benchmark_params_dict: dict = self.benchmark_params.model_dump()

        # Extract required values from the row based on the specified columns
        # feedback_args = [get_nested_value(row, col) for col in required_columns]

        # Append the benchmark parameters dictionary
        feedback_args.append(benchmark_params_dict)

        ret = feedback_fn(*feedback_args)

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
    def __call__(
        self,
        ground_truth: pd.DataFrame,
    ) -> Union[
        List[float], List[Tuple[float]], Tuple[List[float], List[float]]
    ]:
        """Collect the list of generated feedback scores as input to the benchmark aggregation functions
        Note the order of generated scores must be preserved to match the order of the true labels.

        Args:
            ground_truth (pd.DataFrame): ground truth dataset / collection to evaluate the feedback function on

        Returns:
            List[float]: feedback scores after running the benchmark on all entries in ground truth data
        """

        # TODO: instance type check of ground_truth argument + handle groundtruth_impl

        scores = []
        meta_scores = []
        with ThreadPoolExecutor() as executor:
            future_to_index = {}
            index_to_results = {}

            for index, row in ground_truth.iterrows():
                if "expected_chunks" in row:
                    for expected_chunk in row["expected_chunks"]:
                        future = executor.submit(
                            self.run_score_generation_on_single_row,
                            self.feedback_fn,
                            [row["query"], expected_chunk["text"]],
                        )
                        future_to_index[future] = index
                elif "expected_response" in row:
                    future = executor.submit(
                        self.run_score_generation_on_single_row,
                        self.feedback_fn,
                        [row["query"], row["expected_response"]],
                    )
                    future_to_index[future] = index

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    ret = future.result()
                    index_to_results.setdefault(index, []).append(ret)

                except Exception as e:
                    log.error(f"Row generated an exception: {e}")

            # Process results in the original order
            for index in range(len(ground_truth)):
                if index in index_to_results:
                    for ret in index_to_results[index]:
                        if isinstance(ret, float):
                            score = ret
                        else:
                            score, metadata = ret
                            meta_scores.append(metadata)

                        scores.append(score)

        if meta_scores:
            return scores, meta_scores
        else:
            return scores


def create_benchmark_experiment_app(
    app_name: str,
    app_version: str,
    benchmark_experiment: TruBenchmarkExperiment,
    **kwargs,
) -> TruCustomApp:
    """Create a Custom app for special use case: benchmarking feedback functions.

    Args:
        app_name (str): user-defined name of the experiment run.
        app_version (str): user-defined version of the experiment run.
        feedback_fn (Callable): feedback function of interest to perform meta-evaluation on.
        agg_funcs (List[feedback.AggCallable]): list of aggregation functions to compute metrics for the benchmark.
        benchmark_params (Any): parameters for the benchmarking experiment.

    Returns:
        trulens.core.app.TruCustomApp: Custom app wrapper for benchmarking feedback functions.
    """

    return TruCustomApp(
        benchmark_experiment,
        app_name=app_name,
        app_version=app_version,
        feedbacks=benchmark_experiment.f_benchmark_metrics,
        **kwargs,
    )
