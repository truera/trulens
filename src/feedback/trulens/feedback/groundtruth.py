import logging
from typing import (
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import warnings

import numpy as np
import pandas as pd
import pydantic
import scipy.stats as stats
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_auc_score
from trulens.core.utils import imports as import_utils
from trulens.core.utils import pyschema as pyschema_utils
from trulens.core.utils import serial as serial_utils
from trulens.feedback import generated as feedback_generated
from trulens.feedback import llm_provider

with import_utils.OptionalImports(
    messages=import_utils.format_import_errors(
        "bert-score", purpose="measuring BERT Score"
    )
):
    from bert_score import BERTScorer

with import_utils.OptionalImports(
    messages=import_utils.format_import_errors(
        "evaluate", purpose="using certain metrics"
    )
):
    import evaluate

logger = logging.getLogger(__name__)


# TODEP
class GroundTruthAgreement(
    pyschema_utils.WithClassInfo, serial_utils.SerialModel
):
    """Measures Agreement against a Ground Truth."""

    ground_truth: Union[
        List[Dict],
        Callable,
        pd.DataFrame,
        pyschema_utils.FunctionOrMethod,
    ]
    provider: llm_provider.LLMProvider

    # Note: the bert scorer object isn't serializable
    # It's a class member because creating it is expensive
    bert_scorer: object

    ground_truth_imp: Optional[Callable] = pydantic.Field(None, exclude=True)

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True
    )

    def __init__(
        self,
        ground_truth: Union[
            List[Dict], Callable, pd.DataFrame, pyschema_utils.FunctionOrMethod
        ],
        provider: Optional[llm_provider.LLMProvider] = None,
        bert_scorer: Optional["BERTScorer"] = None,
        **kwargs,
    ):
        """Measures Agreement against a Ground Truth.

        Usage 1:
            ```python
            from trulens.feedback import GroundTruthAgreement
            from trulens.providers.openai import OpenAI
            golden_set = [
                {"query": "who invented the lightbulb?", "expected_response": "Thomas Edison"},
                {"query": "¿quien invento la bombilla?", "expected_response": "Thomas Edison"}
            ]
            ground_truth_collection = GroundTruthAgreement(golden_set, provider=OpenAI())
            ```

        Usage 2:
            ```python
            from trulens.feedback import GroundTruthAgreement
            from trulens.providers.openai import OpenAI
            from trulens.core.session import TruSession

            session = TruSession()
            ground_truth_dataset = session.get_ground_truths_by_dataset("hotpotqa") # assuming a dataset "hotpotqa" has been created and persisted in the DB

            ground_truth_collection = GroundTruthAgreement(ground_truth_dataset, provider=OpenAI())
            ```

        Usage 3:
            ```python
            from snowflake.snowpark import Session
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

            snowpark_session = Session.builder.configs(snowflake_connection_parameters).create()

            ground_truth_collection = GroundTruthAgreement(
                ground_truth_imp,
                provider=Cortex(
                    snowpark_session=snowpark_session,
                    model_engine="mistral-7b",
                ),
            )
            ```

        Args:
            ground_truth: A list of query/response pairs or a function, or a
                dataframe containing ground truth dataset, or callable that
                returns a ground truth string given a prompt string.

            provider: The provider to use for
                agreement measures.

            bert_scorer: Internal Usage for
                DB serialization.

        """
        if provider is None:
            warnings.warn(
                "Default provider is being deprecated. Defaulting to OpenAI.",
                DeprecationWarning,
            )
            if not import_utils.is_package_installed(
                "trulens-providers-openai"
            ):
                raise ImportError(
                    "`trulens-providers-openai` package is required for the default OpenAI provider."
                )
            else:
                from trulens.providers.openai import OpenAI

                provider = OpenAI()

        if isinstance(ground_truth, List):
            ground_truth_imp = None
        elif isinstance(ground_truth, pyschema_utils.FunctionOrMethod):
            ground_truth_imp = ground_truth.load()
        elif isinstance(ground_truth, Callable):
            ground_truth_imp = ground_truth
            ground_truth = pyschema_utils.FunctionOrMethod.of_callable(
                ground_truth
            )
        elif isinstance(ground_truth, pd.DataFrame):
            ground_truth_df = ground_truth
            ground_truth = []
            for _, row in ground_truth_df.iterrows():
                entry = row.to_dict()
                ground_truth.append(entry)
            ground_truth_imp = None
        elif isinstance(ground_truth, Dict):
            # Serialized pyschema_utils.FunctionOrMethod?
            ground_truth = pyschema_utils.FunctionOrMethod.model_validate(
                ground_truth
            )
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

    def _find_golden_context_chunks_and_scores(
        self, prompt: str
    ) -> Optional[List[Tuple[str, float]]]:
        if self.ground_truth_imp is not None:
            return self.ground_truth_imp(prompt)

        golden_context_chunks = [
            (
                chunk["text"],
                chunk["expect_score"] if "expect_score" in chunk else 1,
            )
            for qr in self.ground_truth
            for chunk in qr["expected_chunks"]
            if qr["query"] == prompt
        ]
        if golden_context_chunks:
            return golden_context_chunks
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
        Uses OpenAI's Chat GPT Model. A function that measures
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
            float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".
            dict: with key 'ground_truth_response'
        """
        ground_truth_response = self._find_response(prompt)
        if ground_truth_response:
            agreement_txt = self.provider._get_answer_agreement(
                prompt, response, ground_truth_response
            )
            ret = (
                feedback_generated.re_0_10_rating(agreement_txt) / 10,
                dict(ground_truth_response=ground_truth_response),
            )
        else:
            ret = np.nan

        return ret

    def ndcg_at_k(
        self,
        query: str,
        retrieved_context_chunks: List[str],
        relevance_scores: Optional[List[float]] = None,
        k: Optional[int] = None,
    ) -> float:
        """
        Compute NDCG@k for a given query and retrieved context chunks.

        Args:
            query (str): The input query string.
            retrieved_context_chunks (List[str]): List of retrieved context chunks.
            relevance_scores (Optional[List[float]]): Relevance scores for each retrieved chunk.
            k (Optional[int]): Rank position up to which to compute NDCG. If None, compute for all retrieved chunks.

        Returns:
            float: Computed NDCG@k score.
        """
        # Step 1: Find the ground truth context chunks for the given query
        ground_truth_context_chunks_and_scores = (
            self._find_golden_context_chunks_and_scores(query)
        )
        if ground_truth_context_chunks_and_scores:
            k = k or len(
                retrieved_context_chunks
            )  # If k is None, use all retrieved chunks

            # Step 2: Extract ground truth chunks and their relevance scores
            golden_chunks = [
                chunk[0] for chunk in ground_truth_context_chunks_and_scores
            ]
            golden_scores = [
                chunk[1] for chunk in ground_truth_context_chunks_and_scores
            ]

            # Step 3: If relevance scores are provided, sort retrieved chunks by scores in descending order
            if relevance_scores:
                # Zip together retrieved chunks and relevance scores
                retrieved_with_scores = list(
                    zip(retrieved_context_chunks, relevance_scores)
                )
                # Sort by scores in descending order
                retrieved_with_scores.sort(key=lambda x: x[1], reverse=True)
                # Extract the sorted chunks
                retrieved_context_chunks = [
                    chunk for chunk, _ in retrieved_with_scores
                ]

            # Step 4: Create a binary relevance vector for the retrieved chunks based on golden chunks
            rel_scores = [0.0] * len(
                retrieved_context_chunks
            )  # Initialize with 0 relevance for all
            for i, chunk in enumerate(retrieved_context_chunks[:k]):
                if chunk in golden_chunks:
                    index_in_golden = golden_chunks.index(chunk)
                    rel_scores[i] = golden_scores[
                        index_in_golden
                    ]  # Use the true relevance score

            # Step 5: Prepare the ground truth scores as a vector
            # Ideal DCG (IDCG) is calculated by placing all relevant items at the top in descending order
            ideal_golden_scores = sorted(golden_scores, reverse=True)[:k]

            # Step 6: Calculate NDCG@k using sklearn's ndcg_score
            return ndcg_score(
                y_true=np.array([ideal_golden_scores]),
                y_score=np.array([rel_scores[:k]]),  # Consider only top-k items
                k=k,
            )
        else:
            return np.nan

    def precision_at_k(
        self,
        query: str,
        retrieved_context_chunks: List[str],
        relevance_scores: Optional[List[float]] = None,
        k: Optional[int] = None,
    ) -> float:
        """
        Compute Precision@k for a given query and retrieved context chunks, considering tie handling.

        Args:
            query (str): The input query string.
            retrieved_context_chunks (List[str]): List of retrieved context chunks.
            relevance_scores (Optional[List[float]]): Relevance scores for each retrieved chunk.
            k (Optional[int]): Rank position up to which to compute Precision. If None, compute for all retrieved chunks.

        Returns:
            float: Computed Precision@k score.
        """
        ground_truth_context_chunks = (
            self._find_golden_context_chunks_and_scores(query)
        )
        if ground_truth_context_chunks:
            k = k or len(retrieved_context_chunks)

            # Extract ground truth chunks
            golden_chunks = set(
                chunk[0] for chunk in ground_truth_context_chunks
            )

            # Sort retrieved chunks by relevance scores if scores are provided
            if relevance_scores:
                # Get the top-k threshold score (with tie handling)
                sorted_scores = sorted(relevance_scores, reverse=True)
                kth_score = sorted_scores[min(k - 1, len(sorted_scores) - 1)]

                # Include all indices with scores >= kth score
                top_k_indices = [
                    i
                    for i, score in enumerate(relevance_scores)
                    if score >= kth_score
                ]
                retrieved_top_k = [
                    retrieved_context_chunks[i] for i in top_k_indices
                ]
            else:
                # If no relevance scores, use the top-k retrieved chunks as they are
                retrieved_top_k = retrieved_context_chunks[:k]

            # Calculate precision at k with tie handling
            relevant_retrieved = len([
                chunk for chunk in retrieved_top_k if chunk in golden_chunks
            ])
            return (
                relevant_retrieved / len(retrieved_top_k)
                if len(retrieved_top_k) > 0
                else 0.0
            )
        else:
            return np.nan

    def recall_at_k(
        self,
        query: str,
        retrieved_context_chunks: List[str],
        relevance_scores: Optional[List[float]] = None,
        k: Optional[int] = None,
    ) -> float:
        """
        Compute Recall@k for a given query and retrieved context chunks, considering tie handling.

        Args:
            query (str): The input query string.
            retrieved_context_chunks (List[str]): List of retrieved context chunks.
            relevance_scores (Optional[List[float]]): Relevance scores for each retrieved chunk.
            k (Optional[int]): Rank position up to which to compute Recall. If None, compute for all retrieved chunks.

        Returns:
            float: Computed Recall@k score.
        """
        ground_truth_context_chunks = (
            self._find_golden_context_chunks_and_scores(query)
        )
        if ground_truth_context_chunks:
            k = k or len(retrieved_context_chunks)

            # Extract ground truth chunks
            golden_chunks = set(
                chunk[0] for chunk in ground_truth_context_chunks
            )

            # Sort retrieved chunks by relevance scores if scores are provided
            if relevance_scores:
                # Get the top-k threshold score (with tie handling)
                sorted_scores = sorted(relevance_scores, reverse=True)
                kth_score = sorted_scores[min(k - 1, len(sorted_scores) - 1)]

                # Include all indices with scores >= kth score
                top_k_indices = [
                    i
                    for i, score in enumerate(relevance_scores)
                    if score >= kth_score
                ]
                retrieved_top_k = [
                    retrieved_context_chunks[i] for i in top_k_indices
                ]
            else:
                # If no relevance scores, use the top-k retrieved chunks as they are
                retrieved_top_k = retrieved_context_chunks[:k]

            # Calculate recall at k with tie handling
            relevant_retrieved = len([
                chunk for chunk in retrieved_top_k if chunk in golden_chunks
            ])
            return (
                relevant_retrieved / len(golden_chunks)
                if len(golden_chunks) > 0
                else 0.0
            )
        else:
            return np.nan

    def mrr(
        self,
        query: str,
        retrieved_context_chunks: List[str],
        relevance_scores: Optional[List[float]] = None,
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR) for a given query and retrieved context chunks.

        Args:
            query (str): The input query string.
            retrieved_context_chunks (List[str]): List of retrieved context chunks.

        Returns:
            float: Computed MRR score.
        """
        ground_truth_context_chunks = (
            self._find_golden_context_chunks_and_scores(query)
        )
        if ground_truth_context_chunks:
            # Extract ground truth chunks
            golden_chunks = set(
                chunk[0] for chunk in ground_truth_context_chunks
            )

            # Sort the retrieved chunks by relevance scores if provided
            if relevance_scores:
                retrieved_with_scores = list(
                    zip(retrieved_context_chunks, relevance_scores)
                )
                # Sort in descending order of relevance score
                retrieved_with_scores.sort(key=lambda x: x[1], reverse=True)
                retrieved_context_chunks = [
                    chunk for chunk, _ in retrieved_with_scores
                ]

            # Find the rank of the first relevant item in the sorted list
            for i, chunk in enumerate(retrieved_context_chunks):
                if chunk in golden_chunks:
                    return 1 / (
                        i + 1
                    )  # MRR is the reciprocal of the rank (1-based index)

            return 0.0  # No relevant item found
        else:
            return np.nan

    def ir_hit_rate(
        self,
        query: str,
        retrieved_context_chunks: List[str],
        k: Optional[int] = None,
    ) -> float:
        """
        Compute IR Hit Rate (Hit Rate@k) for a given query and retrieved context chunks.

        Args:
            query (str): The input query string.
            retrieved_context_chunks (List[str]): List of retrieved context chunks.
            k (Optional[int]): Rank position up to which to compute Hit Rate. If None, compute for all retrieved chunks.

        Returns:
            float: Computed Hit Rate@k score.
        """
        ground_truth_context_chunks = (
            self._find_golden_context_chunks_and_scores(query)
        )
        if ground_truth_context_chunks:
            k = k or len(retrieved_context_chunks)

            # Extract ground truth chunks
            golden_chunks = set(
                chunk[0] for chunk in ground_truth_context_chunks
            )

            # Calculate hit rate at k (1 if at least one relevant item is retrieved, 0 otherwise)
            return (
                1.0
                if any(
                    chunk in golden_chunks
                    for chunk in retrieved_context_chunks[:k]
                )
                else 0.0
            )
        else:
            return np.nan

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
            float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".
            dict: with key 'ground_truth_response'
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
        """Uses BLEU Score. A function that that measures similarity to ground
        truth using token overlap.

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
            prompt: A text prompt to an agent.

            response: The agent's response to the prompt.

        Returns:
            float: A value between 0 and 1. 0 being "not in agreement" and 1
                being "in agreement".

            dict: with key 'ground_truth_response'
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


class GroundTruthAggregator(
    pyschema_utils.WithClassInfo, serial_utils.SerialModel
):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True, extra="allow"
    )
    """Aggregate benchmarking metrics for ground-truth-based evaluation on feedback functions."""

    true_labels: List[
        Union[int, float]
    ]  # ground truth labels in [0, 1, 0, ...] format
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

    def auc(self, scores: List[float]) -> float:
        """
        Calculate the area under the ROC curve. Can be used for meta-evaluation.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Area under the ROC curve
        """
        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]
        return roc_auc_score(self.true_labels, scores)

    def kendall_tau(self, scores: Union[List[float], List[List]]) -> float:
        """
        Calculate Kendall's tau. Can be used for meta-evaluation.
        Kendall’s tau is a measure of the correspondence between two rankings. Values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement. This is the tau-b version of Kendall’s tau which accounts for ties.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Kendall's tau
        """
        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]
        tau = stats.kendalltau(scores, self.true_labels).statistic
        # The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0.
        # TODO: p_value is unused here
        return tau

    def spearman_correlation(
        self, scores: Union[List[float], List[List]]
    ) -> float:
        """
        Calculate the Spearman correlation. Can be used for meta-evaluation.
        The Spearman correlation coefficient is a nonparametric measure of rank correlation (statistical dependence between the rankings of two variables).

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Spearman correlation

        """
        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]
        x = np.array(scores)
        y = np.array(self.true_labels)

        return stats.spearmanr(x, y).statistic

    def pearson_correlation(
        self, scores: Union[List[float], List[List]]
    ) -> float:
        """
        Calculate the Pearson correlation. Can be used for meta-evaluation.
        The Pearson correlation coefficient is a measure of the linear relationship between two variables.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Pearson correlation

        """
        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]
        x = np.array(scores)
        y = np.array(self.true_labels)

        return stats.pearsonr(x, y)[0]

    def matthews_correlation(
        self, scores: Union[List[float], List[List]]
    ) -> float:
        """
        Calculate the Matthews correlation coefficient. Can be used for meta-evaluation.
        The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Matthews correlation coefficient

        """
        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]
        x = np.array(scores)
        y = np.array(self.true_labels)

        return matthews_corrcoef(y, x)

    def cohens_kappa(
        self, scores: Union[List[float], List[List]], threshold=0.5
    ) -> float:
        """
        Computes Cohen's Kappa score between true labels and predicted scores.

        Parameters:
        - true_labels (list): A list of true labels.
        - scores (list): A list of predicted labels or scores.

        Returns:
        - float: Cohen's Kappa score.
        """
        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]

        if len(self.true_labels) != len(scores):
            raise ValueError(
                "The length of true_labels and scores must be the same."
            )
        #  convert to categorical if necessary
        if any(isinstance(score, float) for score in scores):
            # threshold at 0.5 for binary classification
            scores = [1 if score >= threshold else 0 for score in scores]

        if any(isinstance(label, float) for label in self.true_labels):
            self.true_labels = [
                1 if label >= threshold else 0 for label in self.true_labels
            ]

        kappa = cohen_kappa_score(self.true_labels, scores)
        return kappa

    def recall(self, scores: Union[List[float], List[List]], threshold=0.5):
        """
        Calculates recall given true labels and model-generated scores.

        Parameters:
        - scores (list of float): A list of model-generated scores (0 to 1.0).
        - threshold (float): The threshold to convert scores to binary predictions. Default is 0.5.

        Returns:
        - float: The recall score.
        """

        try:
            if isinstance(scores[0], List):
                scores = [score for score, _ in scores]
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"scores processing failed in Recall aggregator: {e}")
            print(f"scores processing failed in Recall aggregator: {e}")
        # Convert scores to binary predictions based on the threshold
        predictions = [1 if score >= threshold else 0 for score in scores]

        # Calculate true positives and false negatives
        true_positives = sum(
            1
            for true, pred in zip(self.true_labels, predictions)
            if true == 1 and pred == 1
        )
        false_negatives = sum(
            1
            for true, pred in zip(self.true_labels, predictions)
            if true == 1 and pred == 0
        )

        # Handle the case where there are no actual positives to avoid division by zero
        if true_positives + false_negatives == 0:
            return 0.0  # or handle as needed (e.g., return None, raise an exception)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        return recall

    def precision(self, scores: Union[List[float], List[List]], threshold=0.5):
        """
        Calculates precision given true labels and model-generated scores.

        Parameters:
        - scores (list of float): A list of model-generated scores (0 to 1.0).
        - threshold (float): The threshold to convert scores to binary predictions. Default is 0.5.

        Returns:
        - float: The precision score.
        """
        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]

        # Convert scores to binary predictions based on the threshold
        predictions = [1 if score >= threshold else 0 for score in scores]

        # Calculate true positives and false positives
        true_positives = sum(
            1
            for true, pred in zip(self.true_labels, predictions)
            if true == 1 and pred == 1
        )
        false_positives = sum(
            1
            for true, pred in zip(self.true_labels, predictions)
            if true == 0 and pred == 1
        )

        # Handle the case where there are no predicted positives to avoid division by zero
        if true_positives + false_positives == 0:
            return 0.0  # or handle as needed (e.g., return None, raise an exception)

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        return precision

    def f1_score(self, scores: Union[List[float], List[List]], threshold=0.5):
        """
        Calculates the F1 score given true labels and model-generated scores.

        Parameters:
        - scores (list of float): A list of model-generated scores (0 to 1.0).
        - threshold (float): The threshold to convert scores to binary predictions. Default is 0.5.

        Returns:
        - float: The F1 score.
        """
        # Calculate precision and recall
        precision = self.precision(scores, threshold)
        recall = self.recall(scores, threshold)

        # Handle the case where both precision and recall are zero to avoid division by zero
        if precision + recall == 0:
            return 0.0  # or handle as needed (e.g., return None, raise an exception)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def brier_score(self, scores: Union[List[float], List[List]]) -> float:
        """
        assess both calibration and sharpness of the probability estimates
        Args:
            scores (List[float]): relevance scores returned by feedback function
        Returns:
            float: Brier score
        """
        # Brier score is mathematically undefined for empty inputs (division by zero).
        # Return np.nan to accurately represent this undefined state rather than 0.0
        # which would falsely suggest "perfect calibration" with no predictions.
        if not scores:
            return np.nan

        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]
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

    def mae(self, scores: Union[List[float], List[List]]) -> float:
        """
        Calculate the mean absolute error. Can be used for meta-evaluation.

        Args:
            scores (List[float]): scores returned by feedback function

        Returns:
            float: Mean absolute error
        """

        # TODO: refactor this, this is to deal with COT type of response from feedback functions
        if isinstance(scores[0], List):
            scores = [score for score, _ in scores]
            print(f"flatten scores: {scores}")

        return np.mean(np.abs(np.array(scores) - np.array(self.true_labels)))
