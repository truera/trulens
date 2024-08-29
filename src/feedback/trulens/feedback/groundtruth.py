import logging
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pydantic
import scipy.stats as stats
from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_auc_score
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.imports import format_import_errors
from trulens.core.utils.pyschema import FunctionOrMethod
from trulens.core.utils.pyschema import WithClassInfo
from trulens.core.utils.serial import SerialModel
from trulens.feedback.generated import re_0_10_rating
from trulens.feedback.llm_provider import LLMProvider

with OptionalImports(
    messages=format_import_errors("bert-score", purpose="measuring BERT Score")
):
    from bert_score import BERTScorer

with OptionalImports(
    messages=format_import_errors("evaluate", purpose="using certain metrics")
):
    import evaluate

logger = logging.getLogger(__name__)


# TODEP
class GroundTruthAgreement(WithClassInfo, SerialModel):
    """
    Measures Agreement against a Ground Truth.
    """

    ground_truth: Union[List[Dict], Callable, pd.DataFrame, FunctionOrMethod]
    provider: LLMProvider

    # Note: the bert scorer object isn't serializable
    # It's a class member because creating it is expensive
    bert_scorer: object

    ground_truth_imp: Optional[Callable] = pydantic.Field(None, exclude=True)

    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    def __init__(
        self,
        ground_truth: Union[
            List[Dict], Callable, pd.DataFrame, FunctionOrMethod
        ],
        provider: LLMProvider,
        bert_scorer: Optional["BERTScorer"] = None,
        **kwargs,
    ):
        """Measures Agreement against a Ground Truth.

        Usage 1:
        ```
        from trulens.feedback import GroundTruthAgreement
        from trulens.providers.openai import OpenAI
        golden_set = [
            {"query": "who invented the lightbulb?", "expected_response": "Thomas Edison"},
            {"query": "¿quien invento la bombilla?", "expected_response": "Thomas Edison"}
        ]
        ground_truth_collection = GroundTruthAgreement(golden_set, provider=OpenAI())
        ```

        Usage 2:
        from trulens.feedback import GroundTruthAgreement
        from trulens.providers.openai import OpenAI

        session = TruSession()
        ground_truth_dataset = session.get_ground_truths_by_dataset("hotpotqa") # assuming a dataset "hotpotqa" has been created and persisted in the DB

        ground_truth_collection = GroundTruthAgreement(ground_truth_dataset, provider=OpenAI())

        Usage 3:
        ```
        from trulens.feedback import GroundTruthAgreement
        from trulens.providers.cortex import Cortex
        ground_truth_imp = llm_app
        response = llm_app(prompt)

        snowflake_connection_parameters = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ["SNOWFLAKE_DATABASE"],
            "schema": os.environ["SNOWFLAKE_SCHEMA"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        }
        ground_truth_collection = GroundTruthAgreement(
            ground_truth_imp,
            provider=Cortex(
                snowflake.connector.connect(**snowflake_connection_parameters),
                model_engine="mistral-7b",
            ),
        )
        ```

        Args:
            ground_truth (Union[List[Dict], Callable, pd.DataFrame, FunctionOrMethod]): A list of query/response pairs or a function, or a dataframe containing ground truth dataset,
                or callable that returns a ground truth string given a prompt string.
                provider (LLMProvider): The provider to use for agreement measures.
                bert_scorer (Optional[&quot;BERTScorer&quot;], optional): Internal Usage for DB serialization.

        """
        if isinstance(ground_truth, List):
            ground_truth_imp = None
        elif isinstance(ground_truth, FunctionOrMethod):
            ground_truth_imp = ground_truth.load()
        elif isinstance(ground_truth, Callable):
            ground_truth_imp = ground_truth
            ground_truth = FunctionOrMethod.of_callable(ground_truth)
        elif isinstance(ground_truth, pd.DataFrame):
            ground_truth_df = ground_truth
            ground_truth = []
            for _, row in ground_truth_df.iterrows():
                entry = row.to_dict()
                ground_truth.append(entry)
            ground_truth_imp = None
        elif isinstance(ground_truth, Dict):
            # Serialized FunctionOrMethod?
            ground_truth = FunctionOrMethod.model_validate(ground_truth)
            ground_truth_imp = ground_truth.load()
        else:
            raise RuntimeError(
                f"Unhandled ground_truth type: {type(ground_truth)}."
            )

        super().__init__(
            ground_truth=ground_truth,
            ground_truth_imp=ground_truth_imp,
            provider=provider,
            bert_scorer=bert_scorer,
            **kwargs,
        )

    def _find_response(self, prompt: str) -> Optional[str]:
        if self.ground_truth_imp is not None:
            return self.ground_truth_imp(prompt)

        responses = [
            qr["expected_response"]
            for qr in self.ground_truth
            if qr["query"] == prompt
        ]
        if responses:
            return responses[0]
        else:
            return None

    def _find_score(self, prompt: str, response: str) -> Optional[float]:
        if self.ground_truth_imp is not None:
            return self.ground_truth_imp(prompt)

        responses = [
            qr["expected_score"]
            for qr in self.ground_truth
            if qr["query"] == prompt and qr["expected_response"] == response
        ]
        if responses:
            return responses[0]
        else:
            return None

    # TODEP
    def agreement_measure(
        self,
        prompt: str,
        response: str,
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Uses OpenAI's Chat GPT Model. A function that that measures
        similarity to ground truth. A second template is given to Chat GPT
        with a prompt that the original response is correct, and measures
        whether previous Chat GPT's response is similar.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.feedback import GroundTruthAgreement
            from trulens.providers.openai import OpenAI

            golden_set = [
                {"query": "who invented the lightbulb?", "expected_response": "Thomas Edison"},
                {"query": "¿quien invento la bombilla?", "expected_response": "Thomas Edison"}
            ]
            ground_truth_collection = GroundTruthAgreement(golden_set, provider=OpenAI())

            feedback = Feedback(ground_truth_collection.agreement_measure).on_input_output()
            ```
            The `on_input_output()` selector can be changed. See [Feedback Function Guide](https://www.trulens.org/trulens/feedback_function_guide/)

        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.

        Returns:
            - float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".
            - dict: with key 'ground_truth_response'
        """
        ground_truth_response = self._find_response(prompt)
        if ground_truth_response:
            agreement_txt = self.provider._get_answer_agreement(
                prompt, response, ground_truth_response
            )
            ret = (
                re_0_10_rating(agreement_txt) / 10,
                dict(ground_truth_response=ground_truth_response),
            )
        else:
            ret = np.nan

        return ret

    def absolute_error(
        self, prompt: str, response: str, score: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Method to look up the numeric expected score from a golden set and take the difference.

        Primarily used for evaluation of model generated feedback against human feedback

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.feedback import GroundTruthAgreement
            from trulens.providers.bedrock import Bedrock

            golden_set =
            {"query": "How many stomachs does a cow have?", "expected_response": "Cows' diet relies primarily on grazing.", "expected_score": 0.4},
            {"query": "Name some top dental floss brands", "expected_response": "I don't know", "expected_score": 0.8}
            ]

            bedrock = Bedrock(
                model_id="amazon.titan-text-express-v1", region_name="us-east-1"
            )
            ground_truth_collection = GroundTruthAgreement(golden_set, provider=bedrock)

            f_groundtruth = Feedback(ground_truth.absolute_error.on(Select.Record.calls[0].args.args[0]).on(Select.Record.calls[0].args.args[1]).on_output()
            ```

        """

        expected_score = self._find_score(prompt, response)
        if expected_score is not None:
            ret = abs(float(score) - float(expected_score))
            expected_score = (
                "{:.2f}".format(expected_score).rstrip("0").rstrip(".")
            )
        else:
            ret = np.nan
        return ret, {"expected score": expected_score}

    def bert_score(
        self, prompt: str, response: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Uses BERT Score. A function that that measures
        similarity to ground truth using bert embeddings.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.feedback import GroundTruthAgreement
            from trulens.providers.openai import OpenAI
            golden_set = [
                {"query": "who invented the lightbulb?", "expected_response": "Thomas Edison"},
                {"query": "¿quien invento la bombilla?", "expected_response": "Thomas Edison"}
            ]
            ground_truth_collection = GroundTruthAgreement(golden_set, provider=OpenAI())

            feedback = Feedback(ground_truth_collection.bert_score).on_input_output()
            ```
            The `on_input_output()` selector can be changed. See [Feedback Function Guide](https://www.trulens.org/trulens/feedback_function_guide/)


        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.

        Returns:
            - float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".
            - dict: with key 'ground_truth_response'
        """
        if self.bert_scorer is None:
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        ground_truth_response = self._find_response(prompt)
        if ground_truth_response:
            bert_score = self.bert_scorer.score(
                [response], [ground_truth_response]
            )
            ret = (
                bert_score[0].item(),
                dict(ground_truth_response=ground_truth_response),
            )
        else:
            ret = np.nan

        return ret

    # TODEP
    def bleu(
        self, prompt: str, response: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Uses BLEU Score. A function that that measures
        similarity to ground truth using token overlap.

        Example:

            ```python
            from trulens.core import Feedback
            from trulens.feedback import GroundTruthAgreement
            from trulens.providers.openai import OpenAI
            golden_set = [
                {"query": "who invented the lightbulb?", "expected_response": "Thomas Edison"},
                {"query": "¿quien invento la bombilla?", "expected_response": "Thomas Edison"}
            ]
            ground_truth_collection = GroundTruthAgreement(golden_set, provider=OpenAI())

            feedback = Feedback(ground_truth_collection.bleu).on_input_output()
            ```
            The `on_input_output()` selector can be changed. See [Feedback Function Guide](https://www.trulens.org/trulens/feedback_function_guide/)

        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.

        Returns:
            - float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".
            - dict: with key 'ground_truth_response'
        """
        bleu = evaluate.load("bleu")
        ground_truth_response = self._find_response(prompt)
        if ground_truth_response:
            bleu_score = bleu.compute(
                predictions=[response], references=[ground_truth_response]
            )
            ret = (
                bleu_score["bleu"],
                dict(ground_truth_response=ground_truth_response),
            )
        else:
            ret = np.nan

        return ret

    # TODEP
    def rouge(
        self, prompt: str, response: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Uses BLEU Score. A function that that measures
        similarity to ground truth using token overlap.

        Args:
            prompt (str): A text prompt to an agent.
            response (str): The agent's response to the prompt.

        Returns:
            - float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".
            - dict: with key 'ground_truth_response'
        """
        rouge = evaluate.load("rouge")
        ground_truth_response = self._find_response(prompt)
        if ground_truth_response:
            rouge_score = rouge.compute(
                predictions=[response], references=[ground_truth_response]
            )
            ret = (
                rouge_score["rouge1"],
                dict(ground_truth_response=ground_truth_response),
            )
        else:
            ret = np.nan

        return ret

    @property
    def mae(self):
        raise NotImplementedError("`mae` has moved to `GroundTruthAggregator`")


class GroundTruthAggregator(WithClassInfo, SerialModel):
    model_config: ClassVar[dict] = dict(
        arbitrary_types_allowed=True, extra="allow"
    )
    """Aggregate benchmarking metrics for ground-truth-based evaluation on feedback fuctions."""

    true_labels: List[int]  # ground truth labels in [0, 1, 0, ...] format
    custom_agg_funcs: Dict[str, Callable] = pydantic.Field(default_factory=dict)

    k: Optional[int] = (
        None  # top k results to consider in NDCG@k, precision@k, recall@k, etc
    )

    n_bins: int = 5  # number of bins for ECE

    def __init__(
        self, true_labels: List[int], k: Optional[int] = None, **kwargs
    ):
        # TODO: automatically load from IR / benchmarking datasets or just set the DB url for smaller serialization overhead?
        super().__init__(
            true_labels=true_labels, k=k, custom_agg_funcs={}, **kwargs
        )

    def register_custom_agg_func(
        self,
        name: str,
        func: Callable[[List[float], "GroundTruthAggregator"], float],
    ) -> None:
        """Register a custom aggregation function."""
        self.custom_agg_funcs[name] = func

        setattr(self, name, lambda scores: func(scores, self))

    def ndcg_at_k(self, scores: List[float]) -> float:
        """
        NDCG can be used for meta-evaluation of other feedback results, returned as relevance scores.

        Args:
            scores (List[float]): relevance scores returned by feedback function

        Returns:
            float: NDCG@k
        """
        assert self.k is not None, "k must be set for ndcg_at_k"
        relevance_scores = np.array([scores])
        true_labels = np.array([self.true_labels])
        ndcg_values = [ndcg_score(relevance_scores, true_labels, k=self.k)]
        return np.mean(ndcg_values)

    def precision_at_k(self, scores: List[float]) -> float:
        """
        Calculate the precision at K. Can be used for meta-evaluation.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Precision@k
        """
        assert self.k is not None, "k must be set for precision_at_k"
        sorted_scores = sorted(scores, reverse=True)
        kth_score = sorted_scores[min(self.k - 1, len(scores) - 1)]

        # Indices of items with scores >= kth highest score
        top_k_indices = [
            i for i, score in enumerate(scores) if score >= kth_score
        ]

        # Calculate precision
        true_positives = sum(np.take(self.true_labels, top_k_indices))
        return true_positives / len(top_k_indices) if top_k_indices else 0

    def recall_at_k(self, scores: List[float]) -> float:
        """
        Calculate the recall at K. Can be used for meta-evaluation.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Recall@k
        """
        assert self.k is not None, "k must be set for recall_at_k"
        sorted_scores = sorted(scores, reverse=True)
        kth_score = sorted_scores[min(self.k - 1, len(scores) - 1)]

        # Indices of items with scores >= kth highest score
        top_k_indices = [
            i for i, score in enumerate(scores) if score >= kth_score
        ]

        # Calculate recall
        relevant_indices = np.where(self.true_labels)[0]
        hits = sum(idx in top_k_indices for idx in relevant_indices)
        total_relevant = sum(self.true_labels)

        return hits / total_relevant if total_relevant > 0 else 0

    def ir_hit_rate(self, scores: List[float]) -> float:
        """
        Calculate the IR hit rate at top k.
        the proportion of queries for which at least one relevant document is retrieved in the top k results. This metric evaluates whether a relevant document is present among the top k retrieved
        Parameters:
        scores (list or array): The list of scores generated by the model.

        Returns:
        float: The hit rate at top k. Binary 0 or 1.
        """
        assert self.k is not None, "k must be set for ir_hit_rate"

        scores_with_relevance = list(zip(scores, self.true_labels))

        # consistently handle duplicate scores with tie breaking (sorted is stable in python)
        sorted_scores = sorted(
            scores_with_relevance, key=lambda x: x[0], reverse=True
        )

        top_k = sorted_scores[: self.k]

        # Check if there is at least one relevant document in the top k
        hits = any([relevance for _, relevance in top_k])

        return 1.0 if hits else 0.0

    def mrr(self, scores: List[float]) -> float:
        """
        Calculate the mean reciprocal rank. Can be used for meta-evaluation.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Mean reciprocal rank
        """

        reciprocal_ranks = []
        for score, relevance in zip(scores, self.true_labels):
            if relevance == 1:
                reciprocal_ranks.append(1 / (scores.index(score) + 1))
        mean_reciprocal_rank = (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else 0
        )
        return round(mean_reciprocal_rank, 4)

    def auc(self, scores: List[float]) -> float:
        """
        Calculate the area under the ROC curve. Can be used for meta-evaluation.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Area under the ROC curve
        """
        return roc_auc_score(self.true_labels, scores)

    def kendall_tau(self, scores: List[float]) -> float:
        """
        Calculate Kendall's tau. Can be used for meta-evaluation.
        Kendall’s tau is a measure of the correspondence between two rankings. Values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement. This is the tau-b version of Kendall’s tau which accounts for ties.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Kendall's tau
        """
        tau, _p_value = stats.kendalltau(scores, self.true_labels).correlation
        # The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0.
        # TODO: p_value is unused here
        return tau

    def spearman_correlation(self, scores: List[float]) -> float:
        """
        Calculate the Spearman correlation. Can be used for meta-evaluation.
        The Spearman correlation coefficient is a nonparametric measure of rank correlation (statistical dependence between the rankings of two variables).

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Spearman correlation

        """
        x = np.array(scores)
        y = np.array(self.true_labels)

        return stats.spearmanr(x, y).statistic

    def brier_score(self, scores: List[float]) -> float:
        """
        assess both calibration and sharpness of the probability estimates
        Args:
            scores (List[float]): relevance scores returned by feedback function
        Returns:
            float: Brier score
        """
        assert len(scores) == len(self.true_labels)
        brier_score = 0

        for score, truth in zip(scores, self.true_labels):
            brier_score += (score - truth) ** 2

        return brier_score / len(scores)

    def ece(self, score_confidence_pairs: List[Tuple[float]]) -> float:
        """
        Calculate the expected calibration error. Can be used for meta-evaluation.

        Args:
            score_confidence_pairs (List[Tuple[float]]): list of tuples of relevance scores and confidences returned by feedback function

        Returns:
            float: Expected calibration error
        """

        assert len(score_confidence_pairs) == len(self.true_labels)
        scores, confidences = zip(*score_confidence_pairs)

        # uniform binning approach with M number of bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # get predictions from relevance scores
        relevance_scores = np.array(scores)
        confidences = np.array(confidences)
        true_labels = np.array(self.true_labels)
        predicted_labels = (relevance_scores >= 0.5).astype(int)

        # get a boolean list of correct/false predictions
        accuracies = predicted_labels == true_labels

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # determine if sample is in bin m (between bin lower & upper)
            in_bin = np.logical_and(
                confidences > bin_lower, confidences <= bin_upper
            )
            # calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            prob_in_bin = in_bin.mean()

            if prob_in_bin > 0:
                # get the accuracy of bin m: acc(Bm)
                accuracy_in_bin = accuracies[in_bin].mean()
                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = confidences[in_bin].mean()
                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                ece += (
                    np.abs(avg_confidence_in_bin - accuracy_in_bin)
                    * prob_in_bin
                )
        return round(ece, 4)

    def mae(self, scores: List[float]) -> float:
        """
        Calculate the mean absolute error. Can be used for meta-evaluation.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Mean absolute error
        """

        return np.mean(np.abs(np.array(scores) - np.array(self.true_labels)))
