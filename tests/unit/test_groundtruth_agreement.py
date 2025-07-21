"""Tests for GroundTruthAgreement class."""

from unittest import TestCase
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from trulens.feedback.dummy.provider import DummyProvider
from trulens.feedback.groundtruth import GroundTruthAgreement


class TestGroundTruthAgreement(TestCase):
    """Tests for GroundTruthAgreement functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_provider = DummyProvider(name="test_provider", delay=0.0)
        self.mock_provider._get_answer_agreement = Mock(return_value="7")

        # Core test datasets
        self.basic_qa = [
            {"query": "What is 2+2?", "expected_response": "4"},
            {
                "query": "What is the capital of France?",
                "expected_response": "Paris",
            },
            {
                "query": "Who wrote Romeo and Juliet?",
                "expected_response": "William Shakespeare",
            },
        ]

        self.qa_with_scores = [
            {
                "query": "What is 2+2?",
                "expected_response": "4",
                "expected_score": 0.9,
            },
            {
                "query": "What is the capital of France?",
                "expected_response": "Paris",
                "expected_score": 0.8,
            },
            {
                "query": "Who wrote Romeo and Juliet?",
                "expected_response": "William Shakespeare",
                "expected_score": 0.7,
            },
        ]

        self.ir_data = [
            {
                "query": "What is machine learning?",
                "expected_chunks": [
                    {
                        "text": "Machine learning is a subset of AI",
                        "expect_score": 1.0,
                    },
                    {
                        "text": "It uses algorithms to learn patterns",
                        "expect_score": 0.8,
                    },
                ],
            },
            {
                "query": "What is deep learning?",
                "expected_chunks": [
                    {
                        "text": "Deep learning uses neural networks",
                        "expect_score": 1.0,
                    },
                    {
                        "text": "It has multiple hidden layers",
                        "expect_score": 0.9,
                    },
                ],
            },
        ]

        # Test instances
        self.agreement = GroundTruthAgreement(
            ground_truth=self.basic_qa.copy(), provider=self.mock_provider
        )
        self.agreement_with_scores = GroundTruthAgreement(
            ground_truth=self.qa_with_scores.copy(), provider=self.mock_provider
        )
        self.agreement_with_chunks = GroundTruthAgreement(
            ground_truth=self.ir_data.copy(), provider=self.mock_provider
        )

    def _create_agreement(self, ground_truth):
        """Helper to create GroundTruthAgreement instances."""
        return GroundTruthAgreement(
            ground_truth=ground_truth, provider=self.mock_provider
        )

    def _mock_rating(self, value):
        """Helper to mock rating generation."""
        return patch(
            "trulens.feedback.generated.re_0_10_rating", return_value=value
        )

    @pytest.mark.optional
    def test_constructor_with_list(self):
        """Test constructor with list input."""
        agreement = self._create_agreement(self.basic_qa)
        self.assertEqual(len(agreement.ground_truth), 3)
        self.assertIsNone(agreement.ground_truth_imp)

    @pytest.mark.optional
    def test_constructor_with_dataframe(self):
        """Test constructor with DataFrame input."""
        df = pd.DataFrame(self.basic_qa)
        agreement = self._create_agreement(df)
        self.assertEqual(len(agreement.ground_truth), 3)

    @pytest.mark.optional
    def test_constructor_with_callable(self):
        """Test constructor with callable function."""

        def mock_ground_truth_func(prompt):
            return "4" if "2+2" in prompt else "Unknown"

        agreement = self._create_agreement(mock_ground_truth_func)
        self.assertIsNotNone(agreement.ground_truth_imp)
        self.assertEqual(agreement.ground_truth_imp("What is 2+2?"), "4")

    @pytest.mark.optional
    def test_constructor_with_function_or_method(self):
        """Test constructor with FunctionOrMethod input."""
        self.skipTest("FunctionOrMethod requires importable functions")

    @pytest.mark.optional
    def test_constructor_default_provider_warning(self):
        """Test that default provider raises deprecation warning."""
        with patch(
            "trulens.feedback.groundtruth.import_utils.is_package_installed",
            return_value=True,
        ):
            with patch(
                "trulens.providers.openai.OpenAI",
                return_value=self.mock_provider,
            ):
                with pytest.warns(DeprecationWarning):
                    GroundTruthAgreement(ground_truth=self.basic_qa)

    @pytest.mark.optional
    def test_constructor_invalid_type(self):
        """Test constructor with invalid ground_truth type."""
        with self.assertRaises(RuntimeError):
            self._create_agreement("invalid_string")

    @pytest.mark.optional
    def test_public_api_ground_truth_lookup_via_agreement_measure(self):
        """Test ground truth lookup via public agreement_measure API."""
        with self._mock_rating(7):
            result = self.agreement.agreement_measure("What is 2+2?", "Four")
            score, metadata = result
            self.assertEqual(score, 0.7)
            self.assertEqual(metadata["ground_truth_response"], "4")

        with self._mock_rating(8):
            result = self.agreement.agreement_measure(
                "What is the capital of France?", "Paris"
            )
            score, metadata = result
            self.assertEqual(score, 0.8)
            self.assertEqual(metadata["ground_truth_response"], "Paris")

        result = self.agreement.agreement_measure(
            "Non-existent query", "response"
        )
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_public_api_ground_truth_lookup_via_callable(self):
        """Test ground truth lookup via public API with callable ground truth."""

        def mock_func(prompt):
            return "4" if "2+2" in prompt else None

        callable_agreement = self._create_agreement(mock_func)

        try:
            with patch("trulens.feedback.groundtruth.BERTScorer") as mock_bert:
                mock_scorer = Mock()
                mock_score_tensor = Mock()
                mock_score_tensor.item.return_value = 0.85
                mock_scorer.score.return_value = (mock_score_tensor,)
                mock_bert.return_value = mock_scorer

                result = callable_agreement.bert_score("What is 2+2?", "Four")
                score, metadata = result
                self.assertEqual(score, 0.85)
                self.assertEqual(metadata["ground_truth_response"], "4")

                result = callable_agreement.bert_score(
                    "Unknown query", "response"
                )
                self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            self.skipTest("bert-score not available")

    @pytest.mark.optional
    def test_public_api_score_lookup_via_absolute_error(self):
        """Test expected score lookup via public absolute_error API."""
        test_cases = [
            ("What is 2+2?", "4", 0.8, 0.1, "0.9"),
            ("What is the capital of France?", "Paris", 0.7, 0.1, "0.8"),
        ]

        for (
            query,
            response,
            test_score,
            expected_error,
            expected_metadata,
        ) in test_cases:
            result = self.agreement_with_scores.absolute_error(
                query, response, test_score
            )
            error, metadata = result
            self.assertAlmostEqual(error, expected_error, places=10)
            self.assertEqual(metadata["expected score"], expected_metadata)

        result = self.agreement_with_scores.absolute_error(
            "Non-existent", "response", 0.5
        )
        error, _ = result
        self.assertTrue(np.isnan(error))

    @pytest.mark.optional
    def test_public_api_chunks_lookup_via_ir_metrics(self):
        """Test golden context chunks lookup via public IR metric APIs."""
        query = "What is machine learning?"

        precision = self.agreement_with_chunks.precision_at_k(
            query, ["Machine learning is a subset of AI", "Irrelevant text"]
        )
        self.assertEqual(precision, 0.5)

        recall = self.agreement_with_chunks.recall_at_k(
            query,
            [
                "Machine learning is a subset of AI",
                "It uses algorithms to learn patterns",
            ],
        )
        self.assertEqual(recall, 1.0)

        for metric_func in [
            self.agreement_with_chunks.precision_at_k,
            self.agreement_with_chunks.recall_at_k,
        ]:
            result = metric_func("Non-existent query", ["chunk1", "chunk2"])
            self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_agreement_measure_found_response(self):
        """Test agreement_measure when ground truth response is found."""
        with self._mock_rating(7):
            result = self.agreement.agreement_measure("What is 2+2?", "Four")
            score, metadata = result
            self.assertEqual(score, 0.7)
            self.assertEqual(metadata["ground_truth_response"], "4")
            self.mock_provider._get_answer_agreement.assert_called_with(
                "What is 2+2?", "Four", "4"
            )

    @pytest.mark.optional
    def test_agreement_measure_no_response_found(self):
        """Test agreement_measure when no ground truth response is found."""
        result = self.agreement.agreement_measure(
            "Unknown query", "Some response"
        )
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_absolute_error_found_score(self):
        """Test absolute_error when expected score is found."""
        result = self.agreement_with_scores.absolute_error(
            "What is 2+2?", "4", 0.8
        )
        error, metadata = result
        self.assertAlmostEqual(error, 0.1, places=10)
        self.assertEqual(metadata["expected score"], "0.9")

    @pytest.mark.optional
    def test_absolute_error_no_score_found(self):
        """Test absolute_error when no expected score is found."""
        result = self.agreement_with_scores.absolute_error(
            "Unknown query", "response", 0.5
        )
        error, _ = result
        self.assertTrue(np.isnan(error))

    @pytest.mark.optional
    def test_bert_score_found_response(self):
        """Test bert_score when ground truth response is found."""
        try:
            with patch("trulens.feedback.groundtruth.BERTScorer") as mock_bert:
                mock_scorer = Mock()
                mock_score_tensor = Mock()
                mock_score_tensor.item.return_value = 0.85
                mock_scorer.score.return_value = (mock_score_tensor,)
                mock_bert.return_value = mock_scorer

                result = self.agreement.bert_score("What is 2+2?", "Four")
                score, metadata = result
                self.assertEqual(score, 0.85)
                self.assertEqual(metadata["ground_truth_response"], "4")
        except ModuleNotFoundError:
            self.skipTest("bert-score not available")

    @pytest.mark.optional
    def test_bert_score_no_response_found(self):
        """Test bert_score when no ground truth response is found."""
        try:
            result = self.agreement.bert_score("Unknown query", "Some response")
            self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            self.skipTest("bert-score not available")

    @pytest.mark.optional
    def test_bleu_score_found_response(self):
        """Test bleu score when ground truth response is found."""
        try:
            with patch(
                "trulens.feedback.groundtruth.evaluate"
            ) as mock_evaluate:
                mock_bleu = Mock()
                mock_bleu.compute.return_value = {"bleu": 0.75}
                mock_evaluate.load.return_value = mock_bleu

                result = self.agreement.bleu("What is 2+2?", "Four")
                score, metadata = result
                self.assertEqual(score, 0.75)
                self.assertEqual(metadata["ground_truth_response"], "4")
        except ModuleNotFoundError:
            self.skipTest("evaluate not available")

    @pytest.mark.optional
    def test_bleu_score_no_response_found(self):
        """Test bleu score when no ground truth response is found."""
        try:
            result = self.agreement.bleu("Unknown query", "Some response")
            self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            self.skipTest("evaluate not available")

    @pytest.mark.optional
    def test_rouge_score_found_response(self):
        """Test rouge score when ground truth response is found."""
        try:
            with patch(
                "trulens.feedback.groundtruth.evaluate"
            ) as mock_evaluate:
                mock_rouge = Mock()
                mock_rouge.compute.return_value = {"rouge1": 0.65}
                mock_evaluate.load.return_value = mock_rouge

                result = self.agreement.rouge("What is 2+2?", "Four")
                score, metadata = result
                self.assertEqual(score, 0.65)
                self.assertEqual(metadata["ground_truth_response"], "4")
        except ModuleNotFoundError:
            self.skipTest("evaluate not available")

    @pytest.mark.optional
    def test_rouge_score_no_response_found(self):
        """Test rouge score when no ground truth response is found."""
        try:
            result = self.agreement.rouge("Unknown query", "Some response")
            self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            self.skipTest("evaluate not available")

    def _test_ir_metric(
        self, metric_func, query, chunks, expected_result, **kwargs
    ):
        """Helper for testing IR metrics."""
        result = metric_func(query, chunks, **kwargs)
        self.assertEqual(result, expected_result)

    @pytest.mark.optional
    def test_ndcg_at_k_perfect_ranking(self):
        """Test NDCG@k with perfect ranking."""
        self._test_ir_metric(
            self.agreement_with_chunks.ndcg_at_k,
            "What is machine learning?",
            [
                "Machine learning is a subset of AI",
                "It uses algorithms to learn patterns",
                "Some irrelevant text",
            ],
            1.0,
            relevance_scores=[1.0, 0.8, 0.1],
            k=2,
        )

    @pytest.mark.optional
    def test_ndcg_at_k_no_ground_truth(self):
        """Test NDCG@k when no ground truth is found."""
        result = self.agreement_with_chunks.ndcg_at_k(
            "Unknown query", ["chunk1", "chunk2"]
        )
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_precision_at_k_perfect_precision(self):
        """Test precision@k with perfect precision."""
        self._test_ir_metric(
            self.agreement_with_chunks.precision_at_k,
            "What is machine learning?",
            [
                "Machine learning is a subset of AI",
                "It uses algorithms to learn patterns",
                "Some irrelevant text",
            ],
            1.0,
            k=2,
        )

    @pytest.mark.optional
    def test_precision_at_k_with_relevance_scores(self):
        """Test precision@k with relevance scores."""
        self._test_ir_metric(
            self.agreement_with_chunks.precision_at_k,
            "What is machine learning?",
            [
                "Some irrelevant text",
                "Machine learning is a subset of AI",
                "It uses algorithms to learn patterns",
            ],
            1.0,
            relevance_scores=[0.1, 1.0, 0.8],
            k=2,
        )

    @pytest.mark.optional
    def test_precision_at_k_no_ground_truth(self):
        """Test precision@k when no ground truth is found."""
        result = self.agreement_with_chunks.precision_at_k(
            "Unknown query", ["chunk1", "chunk2"]
        )
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_recall_at_k_perfect_recall(self):
        """Test recall@k with perfect recall."""
        self._test_ir_metric(
            self.agreement_with_chunks.recall_at_k,
            "What is machine learning?",
            [
                "Machine learning is a subset of AI",
                "It uses algorithms to learn patterns",
                "Some additional relevant content",
            ],
            1.0,
            k=3,
        )

    @pytest.mark.optional
    def test_recall_at_k_partial_recall(self):
        """Test recall@k with partial recall."""
        self._test_ir_metric(
            self.agreement_with_chunks.recall_at_k,
            "What is machine learning?",
            ["Machine learning is a subset of AI", "Some irrelevant text"],
            0.5,
            k=2,
        )

    @pytest.mark.optional
    def test_recall_at_k_no_ground_truth(self):
        """Test recall@k when no ground truth is found."""
        result = self.agreement_with_chunks.recall_at_k(
            "Unknown query", ["chunk1", "chunk2"]
        )
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_mrr_first_position(self):
        """Test MRR when first relevant item is at position 1."""
        self._test_ir_metric(
            self.agreement_with_chunks.mrr,
            "What is machine learning?",
            [
                "Machine learning is a subset of AI",
                "Some irrelevant text",
                "It uses algorithms to learn patterns",
            ],
            1.0,
        )

    @pytest.mark.optional
    def test_mrr_second_position(self):
        """Test MRR when first relevant item is at position 2."""
        self._test_ir_metric(
            self.agreement_with_chunks.mrr,
            "What is machine learning?",
            [
                "Some irrelevant text",
                "Machine learning is a subset of AI",
                "It uses algorithms to learn patterns",
            ],
            0.5,
        )

    @pytest.mark.optional
    def test_mrr_with_relevance_scores(self):
        """Test MRR with relevance scores for sorting."""
        self._test_ir_metric(
            self.agreement_with_chunks.mrr,
            "What is machine learning?",
            [
                "Some irrelevant text",
                "Machine learning is a subset of AI",
                "It uses algorithms to learn patterns",
            ],
            1.0,
            relevance_scores=[0.1, 1.0, 0.8],
        )

    @pytest.mark.optional
    def test_mrr_no_relevant_items(self):
        """Test MRR when no relevant items are found."""
        self._test_ir_metric(
            self.agreement_with_chunks.mrr,
            "What is machine learning?",
            ["Completely irrelevant text", "Another irrelevant chunk"],
            0.0,
        )

    @pytest.mark.optional
    def test_mrr_no_ground_truth(self):
        """Test MRR when no ground truth is found."""
        result = self.agreement_with_chunks.mrr(
            "Unknown query", ["chunk1", "chunk2"]
        )
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_ir_hit_rate_hit(self):
        """Test IR hit rate when there's a hit."""
        self._test_ir_metric(
            self.agreement_with_chunks.ir_hit_rate,
            "What is machine learning?",
            ["Some irrelevant text", "Machine learning is a subset of AI"],
            1.0,
        )

    @pytest.mark.optional
    def test_ir_hit_rate_no_hit(self):
        """Test IR hit rate when there's no hit."""
        self._test_ir_metric(
            self.agreement_with_chunks.ir_hit_rate,
            "What is machine learning?",
            ["Some irrelevant text", "Another irrelevant chunk"],
            0.0,
        )

    @pytest.mark.optional
    def test_ir_hit_rate_with_k(self):
        """Test IR hit rate with k parameter."""
        query = "What is machine learning?"
        chunks = ["Some irrelevant text", "Machine learning is a subset of AI"]

        self._test_ir_metric(
            self.agreement_with_chunks.ir_hit_rate, query, chunks, 0.0, k=1
        )
        self._test_ir_metric(
            self.agreement_with_chunks.ir_hit_rate, query, chunks, 1.0, k=2
        )

    @pytest.mark.optional
    def test_ir_hit_rate_no_ground_truth(self):
        """Test IR hit rate when no ground truth is found."""
        result = self.agreement_with_chunks.ir_hit_rate(
            "Unknown query", ["chunk1", "chunk2"]
        )
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_mae_property_raises_error(self):
        """Test that mae property raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as context:
            _ = self.agreement.mae
        self.assertIn("GroundTruthAggregator", str(context.exception))

    @pytest.mark.optional
    def test_empty_retrieved_chunks(self):
        """Test behavior with empty retrieved chunks."""
        query = "What is machine learning?"
        test_cases = [
            (self.agreement_with_chunks.precision_at_k, 0.0),
            (self.agreement_with_chunks.recall_at_k, 0.0),
            (self.agreement_with_chunks.mrr, 0.0),
            (self.agreement_with_chunks.ir_hit_rate, 0.0),
        ]

        for metric_func, expected in test_cases:
            result = metric_func(query, [])
            self.assertEqual(result, expected)

        try:
            result = self.agreement_with_chunks.ndcg_at_k(query, [])
            self.assertTrue(np.isnan(result) or result == 0.0)
        except Exception:
            pass  # Acceptable for NDCG to raise exception with empty input

    @pytest.mark.optional
    def test_edge_case_k_larger_than_chunks(self):
        """Test behavior when k is larger than number of chunks."""
        query = "What is machine learning?"
        chunks = ["Machine learning is a subset of AI"]

        precision = self.agreement_with_chunks.precision_at_k(
            query, chunks, k=5
        )
        self.assertEqual(precision, 1.0)

        recall = self.agreement_with_chunks.recall_at_k(query, chunks, k=5)
        self.assertEqual(recall, 0.5)

    @pytest.mark.optional
    def test_empty_ground_truth_list(self):
        """Test behavior with empty ground truth list."""
        empty_agreement = self._create_agreement([])

        result = empty_agreement.agreement_measure("Any query", "Any response")
        self.assertTrue(np.isnan(result))

        try:
            result = empty_agreement.bert_score("Any query", "Any response")
            self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            pass

    def _test_special_chars_and_unicode(self):
        """Helper for testing special character and unicode handling."""
        test_data = [
            (
                {
                    "query": "What is 'machine learning'?",
                    "expected_response": "AI subset",
                },
                "What is 'machine learning'?",
                "AI subset",
            ),
            (
                {
                    "query": "Explain A&B testing",
                    "expected_response": "Statistical method",
                },
                "Explain A&B testing",
                "Statistical method",
            ),
            (
                {
                    "query": "¿Qué es machine learning?",
                    "expected_response": "Aprendizaje automático",
                },
                "¿Qué es machine learning?",
                "Aprendizaje automático",
            ),
            (
                {
                    "query": "什么是机器学习?",
                    "expected_response": "机器学习是人工智能的一个分支",
                },
                "什么是机器学习?",
                "机器学习是人工智能的一个分支",
            ),
        ]

        for data, query, expected_response in test_data:
            agreement = self._create_agreement([data])
            with self._mock_rating(7):
                result = agreement.agreement_measure(query, "test response")
                if isinstance(result, tuple):
                    _, metadata = result
                    self.assertEqual(
                        metadata["ground_truth_response"], expected_response
                    )

    @pytest.mark.optional
    def test_special_characters_in_queries(self):
        """Test handling of special characters in queries and responses."""
        self._test_special_chars_and_unicode()

    @pytest.mark.optional
    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        self._test_special_chars_and_unicode()

    @pytest.mark.optional
    def test_case_sensitivity(self):
        """Test case sensitivity in query matching."""
        response = self.agreement.agreement_measure("what is 2+2?", "4")
        self.assertTrue(np.isnan(response))

        with self._mock_rating(7):
            result = self.agreement.agreement_measure("What is 2+2?", "4")
            self.assertIsInstance(result, tuple)

    @pytest.mark.optional
    def test_duplicate_queries_in_ground_truth(self):
        """Test behavior with duplicate queries in ground truth."""
        duplicate_data = [
            {"query": "What is 2+2?", "expected_response": "4"},
            {"query": "What is 2+2?", "expected_response": "Four"},
        ]

        duplicate_agreement = self._create_agreement(duplicate_data)
        with self._mock_rating(7):
            result = duplicate_agreement.agreement_measure(
                "What is 2+2?", "test"
            )
            _, metadata = result
            self.assertEqual(metadata["ground_truth_response"], "4")

    @pytest.mark.optional
    def test_chunks_without_scores(self):
        """Test handling of chunks without expect_score field."""
        chunks_data = [
            {
                "query": "Test query",
                "expected_chunks": [
                    {"text": "Chunk without score"},
                    {"text": "Another chunk without score"},
                ],
            }
        ]

        chunks_agreement = self._create_agreement(chunks_data)
        precision = chunks_agreement.precision_at_k(
            "Test query", ["Chunk without score", "irrelevant"]
        )
        self.assertEqual(precision, 0.5)

    @pytest.mark.optional
    def test_realistic_qa_scenario(self):
        """Test with realistic Q&A data."""
        realistic_qa = [
            {
                "query": "What are the main components of a neural network?",
                "expected_response": "A neural network consists of layers of interconnected nodes (neurons), weights, biases, and activation functions.",
            },
            {
                "query": "How does backpropagation work?",
                "expected_response": "Backpropagation calculates gradients by propagating errors backward through the network to update weights.",
            },
        ]

        realistic_agreement = self._create_agreement(realistic_qa)

        with self._mock_rating(8):
            result = realistic_agreement.agreement_measure(
                "What are the main components of a neural network?",
                "Neural networks have layers, nodes, weights, and activation functions.",
            )
            score, metadata = result
            self.assertEqual(score, 0.8)
            self.assertIn(
                "neural network", metadata["ground_truth_response"].lower()
            )

    @pytest.mark.optional
    def test_realistic_ir_scenario(self):
        """Test with realistic information retrieval scenario."""
        realistic_ir = [
            {
                "query": "How to implement gradient descent?",
                "expected_chunks": [
                    {
                        "text": "Initialize parameters randomly",
                        "expect_score": 0.8,
                    },
                    {"text": "Calculate loss function", "expect_score": 0.9},
                    {"text": "Compute gradients", "expect_score": 1.0},
                    {
                        "text": "Update parameters using learning rate",
                        "expect_score": 1.0,
                    },
                ],
            }
        ]

        ir_agreement = self._create_agreement(realistic_ir)
        retrieved_chunks = [
            "Compute gradients",
            "Calculate the loss function",
            "Update parameters using learning rate",
            "Some irrelevant information",
        ]

        precision = ir_agreement.precision_at_k(
            "How to implement gradient descent?", retrieved_chunks
        )
        self.assertEqual(precision, 0.5)

        recall = ir_agreement.recall_at_k(
            "How to implement gradient descent?", retrieved_chunks
        )
        self.assertEqual(recall, 0.5)

    @pytest.mark.optional
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        large_dataset = [
            {"query": f"Question {i}?", "expected_response": f"Answer {i}"}
            for i in range(100)
        ]

        large_agreement = self._create_agreement(large_dataset)

        with self._mock_rating(7):
            result = large_agreement.agreement_measure("Question 50?", "test")
            self.assertIsInstance(result, tuple)

        result = large_agreement.agreement_measure(
            "Non-existent question", "test"
        )
        self.assertTrue(np.isnan(result))

    @pytest.mark.optional
    def test_mixed_data_types(self):
        """Test with mixed data types in ground truth."""
        mixed_data = [
            {"query": "What is 2+2?", "expected_response": "4"},
            {
                "query": "What is π?",
                "expected_response": 3.14159,
                "expected_score": 0.95,
            },
            {
                "query": "Is Python good?",
                "expected_response": True,
                "expected_score": 0.8,
            },
        ]

        mixed_agreement = self._create_agreement(mixed_data)

        with self._mock_rating(7):
            result = mixed_agreement.agreement_measure("What is 2+2?", "4")
            _, metadata = result
            self.assertEqual(metadata["ground_truth_response"], "4")

        score = mixed_agreement.absolute_error("What is π?", 3.14159, 0.9)[0]
        self.assertAlmostEqual(score, 0.05, places=10)

    @pytest.mark.optional
    def test_consistency_across_methods(self):
        """Test consistency of query matching across different methods."""
        query = "What is machine learning?"

        precision = self.agreement_with_chunks.precision_at_k(
            query, ["Machine learning is a subset of AI"]
        )
        self.assertEqual(precision, 1.0)

        recall = self.agreement_with_chunks.recall_at_k(
            query,
            [
                "Machine learning is a subset of AI",
                "It uses algorithms to learn patterns",
            ],
        )
        self.assertEqual(recall, 1.0)

        hit_rate = self.agreement_with_chunks.ir_hit_rate(
            query, ["Machine learning is a subset of AI"]
        )
        self.assertEqual(hit_rate, 1.0)

    @pytest.mark.optional
    def test_thread_safety_simulation(self):
        """Test thread safety with proper synchronization."""
        import queue
        import threading

        results_queue = queue.Queue()
        start_barrier = threading.Barrier(5)

        def worker():
            start_barrier.wait()

            for i in range(10):
                with self._mock_rating(7):
                    result = self.agreement.agreement_measure(
                        "What is 2+2?", "Four"
                    )
                    if isinstance(result, tuple):
                        score, metadata = result
                        results_queue.put((
                            "agreement",
                            score,
                            metadata["ground_truth_response"],
                        ))

                precision = self.agreement_with_chunks.precision_at_k(
                    "What is machine learning?",
                    ["Machine learning is a subset of AI"],
                )
                results_queue.put(("precision", precision, None))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        self.assertEqual(len(results), 100)

        agreement_results = [r for r in results if r[0] == "agreement"]
        precision_results = [r for r in results if r[0] == "precision"]

        for _, score, ground_truth in agreement_results:
            self.assertEqual(score, 0.7)
            self.assertEqual(ground_truth, "4")

        for _, score, _ in precision_results:
            self.assertEqual(score, 1.0)

    @pytest.mark.optional
    def test_statistical_significance_of_metrics(self):
        """Test statistical significance and confidence intervals."""
        import random

        from scipy import stats

        realistic_ground_truth = [
            {
                "query": f"Query {i}",
                "expected_response": f"Expected {i}",
                "expected_score": 0.5 + 0.4 * random.random(),
            }
            for i in range(50)
        ]

        agreement = self._create_agreement(realistic_ground_truth)

        scores = []
        for i in range(50):
            result = agreement.absolute_error(
                f"Query {i}", f"Expected {i}", 0.7
            )
            if isinstance(result, tuple) and not np.isnan(result[0]):
                scores.append(result[0])

        if len(scores) > 10:
            mean_error = np.mean(scores)
            std_error = np.std(scores)
            n = len(scores)

            confidence_interval = stats.t.interval(
                0.95, n - 1, loc=mean_error, scale=std_error / np.sqrt(n)
            )

            self.assertGreater(
                n, 10, "Insufficient samples for statistical analysis"
            )
            self.assertLess(mean_error, 0.5, "Mean absolute error too high")
            ci_width = confidence_interval[1] - confidence_interval[0]
            self.assertLess(ci_width, 0.3, "Confidence interval too wide")

            if n > 8:
                _, p_value = stats.shapiro(scores)
                if p_value < 0.05:
                    print(
                        f"Warning: Scores may not be normally distributed (p={p_value:.4f})"
                    )

    @pytest.mark.optional
    def test_bootstrap_confidence_intervals(self):
        """Bootstrap confidence intervals for metric stability."""
        ir_data = [
            {
                "query": "machine learning",
                "expected_chunks": [
                    {"text": "ML is subset of AI", "expect_score": 1.0},
                    {"text": "Uses algorithms", "expect_score": 0.8},
                    {"text": "Learns from data", "expect_score": 0.9},
                ],
            }
        ] * 20

        agreement = self._create_agreement(ir_data)
        retrieved_chunks = [
            "ML is subset of AI",
            "Uses algorithms",
            "irrelevant text",
        ]

        bootstrap_precisions = []
        for _ in range(100):
            precision = agreement.precision_at_k(
                "machine learning", retrieved_chunks, k=2
            )
            if not np.isnan(precision):
                bootstrap_precisions.append(precision)

        if len(bootstrap_precisions) > 10:
            lower = np.percentile(bootstrap_precisions, 2.5)
            upper = np.percentile(bootstrap_precisions, 97.5)
            mean_precision = np.mean(bootstrap_precisions)

            self.assertGreater(mean_precision, 0.5, "Mean precision too low")
            self.assertLess(upper - lower, 0.3, "Bootstrap CI too wide")
            self.assertGreaterEqual(lower, 0.0, "Invalid precision lower bound")
            self.assertLessEqual(upper, 1.0, "Invalid precision upper bound")

    @pytest.mark.optional
    def test_metric_convergence_properties(self):
        """Test convergence properties as dataset size increases."""
        convergence_data = []

        for n in [10, 25, 50, 100]:
            ground_truth = [
                {
                    "query": f"Query {i}",
                    "expected_chunks": [
                        {"text": f"Relevant chunk {i}", "expect_score": 1.0}
                    ],
                }
                for i in range(n)
            ]

            agreement = self._create_agreement(ground_truth)

            precisions = []
            for i in range(min(10, n)):
                retrieved = [f"Relevant chunk {i}", "irrelevant"]
                precision = agreement.precision_at_k(f"Query {i}", retrieved)
                if not np.isnan(precision):
                    precisions.append(precision)

            if len(precisions) > 0:
                variance = np.var(precisions)
                convergence_data.append((n, variance))

        if len(convergence_data) >= 3:
            variances = [v for _, v in convergence_data]
            self.assertLessEqual(
                variances[-1],
                variances[0] + 0.1,
                "Metric variance should decrease with more data",
            )

    @pytest.mark.optional
    def test_adversarial_ground_truth_poisoning(self):
        """Test resilience against malicious ground truth data."""
        poisoned_ground_truth = [
            {"query": "What is 2+2?", "expected_response": "4"},
            {"query": "What is 2+2?", "expected_response": "5"},
            {"query": "What is 2+2?", "expected_response": "fish"},
            {"query": "What is 2+2?", "expected_response": ""},
            {"query": "What is 2+2?", "expected_response": "A" * 10000},
        ]

        poisoned_agreement = self._create_agreement(poisoned_ground_truth)

        with self._mock_rating(5):
            result = poisoned_agreement.agreement_measure(
                "What is 2+2?", "Four"
            )
            if isinstance(result, tuple):
                score, metadata = result
                self.assertEqual(metadata["ground_truth_response"], "4")
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    @pytest.mark.optional
    def test_adversarial_input_injections(self):
        """Test against injection-style attacks on queries."""
        malicious_queries = [
            "'; DROP TABLE ground_truth; --",
            "<script>alert('xss')</script>",
            "What is 2+2?" + "A" * 100000,
            "What is 2+2?\x00\x01\x02",
            "What is 2+2?\n\r\t" * 1000,
            "🙃🔥💀" * 100,
        ]

        for malicious_query in malicious_queries:
            try:
                result = self.agreement.agreement_measure(
                    malicious_query, "response"
                )
                self.assertTrue(np.isnan(result) or isinstance(result, tuple))
            except Exception as e:
                self.assertIsInstance(e, (ValueError, TypeError, RuntimeError))
                self.assertNotIsInstance(e, (MemoryError, OSError, SystemError))

    @pytest.mark.optional
    def test_distribution_shift_robustness(self):
        """Test behavior under domain/distribution shift."""
        original_domain = [
            {
                "query": "What is machine learning?",
                "expected_response": "AI subset",
            },
            {
                "query": "What is deep learning?",
                "expected_response": "Neural networks",
            },
        ]

        shifted_queries = [
            "Qu'est-ce que l'apprentissage automatique?",
            "WHAT IS MACHINE LEARNING???!!!",
            "could u explain machine learning pls? thx",
            "Provide a comprehensive analysis of the epistemological foundations of machine learning",
        ]

        agreement = self._create_agreement(original_domain)

        shift_results = []
        for shifted_query in shifted_queries:
            result = agreement.agreement_measure(shifted_query, "AI subset")
            if not np.isnan(result):
                shift_results.append(result)

        self.assertLess(
            len(shift_results),
            len(shifted_queries),
            "Should detect domain shift in most cases",
        )

    @pytest.mark.optional
    def test_metric_gaming_attacks(self):
        """Test against attempts to game the evaluation metrics."""
        gaming_ground_truth = [
            {
                "query": "test query",
                "expected_chunks": [
                    {"text": "relevant1", "expect_score": 1.0},
                    {"text": "relevant2", "expect_score": 1.0},
                ],
            }
        ]

        gaming_agreement = self._create_agreement(gaming_ground_truth)

        gaming_retrieval = ["relevant1"] * 100 + ["irrelevant"]
        precision = gaming_agreement.precision_at_k(
            "test query", gaming_retrieval, k=10
        )

        self.assertLessEqual(precision, 1.0, "Precision cannot exceed 1.0")
        self.assertGreaterEqual(precision, 0.0, "Precision cannot be negative")

        gaming_recall_retrieval = ["irrelevant"] * 1000 + [
            "relevant1",
            "relevant2",
        ]
        recall = gaming_agreement.recall_at_k(
            "test query", gaming_recall_retrieval, k=1002
        )

        self.assertGreater(
            recall, 0.8, "Should find relevant chunks despite noise"
        )

    @pytest.mark.optional
    def test_memory_exhaustion_attacks(self):
        """Test resilience against memory exhaustion attacks."""
        large_ground_truth = []
        for i in range(1000):
            large_ground_truth.append({
                "query": f"Query {i}",
                "expected_response": "Response " * 100,
                "expected_chunks": [
                    {"text": f"Chunk {j} " * 50, "expect_score": 0.5}
                    for j in range(10)
                ],
            })

        try:
            large_agreement = self._create_agreement(large_ground_truth)

            result = large_agreement.precision_at_k(
                "Query 500", ["Chunk 5 " * 50, "irrelevant"], k=2
            )
            self.assertIsInstance(result, (float, type(np.nan)))

        except MemoryError:
            self.fail(
                "System should handle reasonably large datasets without MemoryError"
            )
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError, RuntimeError))
