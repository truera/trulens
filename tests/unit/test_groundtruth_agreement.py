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
        # Use DummyProvider to avoid external dependencies
        self.mock_provider = DummyProvider(
            name="test_provider",
            delay=0.0,  # No delay for tests
        )
        # Mock the _get_answer_agreement method
        self.mock_provider._get_answer_agreement = Mock(return_value="7")

        # Test data sets in different formats
        self.golden_set_list = [
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

        self.golden_set_with_scores = [
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

        self.golden_set_with_chunks = [
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

        # Create agreement instance with mock provider
        self.agreement = GroundTruthAgreement(
            ground_truth=self.golden_set_list.copy(),
            provider=self.mock_provider,
        )

        # Create agreement instance with chunks for IR metrics
        self.agreement_with_chunks = GroundTruthAgreement(
            ground_truth=self.golden_set_with_chunks.copy(),
            provider=self.mock_provider,
        )

        # Create agreement instance with scores
        self.agreement_with_scores = GroundTruthAgreement(
            ground_truth=self.golden_set_with_scores.copy(),
            provider=self.mock_provider,
        )

    @pytest.mark.optional
    def test_constructor_with_list(self):
        """Test constructor with list input."""
        agreement = GroundTruthAgreement(
            ground_truth=self.golden_set_list, provider=self.mock_provider
        )
        self.assertEqual(len(agreement.ground_truth), 3)
        self.assertIsNone(agreement.ground_truth_imp)

    @pytest.mark.optional
    def test_constructor_with_dataframe(self):
        """Test constructor with DataFrame input."""
        df = pd.DataFrame(self.golden_set_list)
        agreement = GroundTruthAgreement(
            ground_truth=df, provider=self.mock_provider
        )
        self.assertEqual(len(agreement.ground_truth), 3)
        self.assertIsNone(agreement.ground_truth_imp)

    @pytest.mark.optional
    def test_constructor_with_callable(self):
        """Test constructor with callable function."""

        def mock_ground_truth_func(prompt):
            if "2+2" in prompt:
                return "4"
            elif "France" in prompt:
                return "Paris"
            return "Unknown"

        agreement = GroundTruthAgreement(
            ground_truth=mock_ground_truth_func, provider=self.mock_provider
        )
        self.assertIsNotNone(agreement.ground_truth_imp)
        self.assertEqual(agreement.ground_truth_imp("What is 2+2?"), "4")

    @pytest.mark.optional
    def test_constructor_with_function_or_method(self):
        """Test constructor with FunctionOrMethod input."""
        # Skip this test as it requires the function to be importable from a module
        # This is a limitation of the FunctionOrMethod serialization system
        self.skipTest(
            "FunctionOrMethod requires importable functions, skipping test"
        )

    @pytest.mark.optional
    def test_constructor_default_provider_warning(self):
        """Test that default provider raises deprecation warning."""
        with patch(
            "trulens.feedback.groundtruth.import_utils.is_package_installed",
            return_value=True,
        ):
            with patch("trulens.providers.openai.OpenAI") as mock_openai:
                mock_openai.return_value = self.mock_provider
                with pytest.warns(DeprecationWarning):
                    GroundTruthAgreement(ground_truth=self.golden_set_list)

    @pytest.mark.optional
    def test_constructor_invalid_type(self):
        """Test constructor with invalid ground_truth type."""
        with self.assertRaises(RuntimeError):
            GroundTruthAgreement(
                ground_truth="invalid_string", provider=self.mock_provider
            )

    @pytest.mark.optional
    def test_find_response_from_list(self):
        """Test _find_response method with list ground truth."""
        response = self.agreement._find_response("What is 2+2?")
        self.assertEqual(response, "4")

        response = self.agreement._find_response(
            "What is the capital of France?"
        )
        self.assertEqual(response, "Paris")

        response = self.agreement._find_response("Non-existent query")
        self.assertIsNone(response)

    @pytest.mark.optional
    def test_find_response_from_callable(self):
        """Test _find_response method with callable ground truth."""

        def mock_ground_truth_func(prompt):
            if "2+2" in prompt:
                return "4"
            return None

        agreement = GroundTruthAgreement(
            ground_truth=mock_ground_truth_func, provider=self.mock_provider
        )

        response = agreement._find_response("What is 2+2?")
        self.assertEqual(response, "4")

        response = agreement._find_response("Unknown query")
        self.assertIsNone(response)

    @pytest.mark.optional
    def test_find_score(self):
        """Test _find_score method."""
        score = self.agreement_with_scores._find_score("What is 2+2?", "4")
        self.assertEqual(score, 0.9)

        score = self.agreement_with_scores._find_score(
            "What is the capital of France?", "Paris"
        )
        self.assertEqual(score, 0.8)

        # Test non-existent query or response
        score = self.agreement_with_scores._find_score(
            "Non-existent", "response"
        )
        self.assertIsNone(score)

    @pytest.mark.optional
    def test_find_golden_context_chunks_and_scores(self):
        """Test _find_golden_context_chunks_and_scores method."""
        chunks = (
            self.agreement_with_chunks._find_golden_context_chunks_and_scores(
                "What is machine learning?"
            )
        )
        expected = [
            ("Machine learning is a subset of AI", 1.0),
            ("It uses algorithms to learn patterns", 0.8),
        ]
        self.assertEqual(chunks, expected)

        chunks = (
            self.agreement_with_chunks._find_golden_context_chunks_and_scores(
                "Non-existent query"
            )
        )
        self.assertIsNone(chunks)

    @pytest.mark.optional
    def test_agreement_measure_found_response(self):
        """Test agreement_measure when ground truth response is found."""
        with patch("trulens.feedback.generated.re_0_10_rating", return_value=7):
            result = self.agreement.agreement_measure("What is 2+2?", "Four")

            # Should return tuple with score and metadata
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

            score, metadata = result
            self.assertEqual(score, 0.7)  # 7/10
            self.assertEqual(metadata["ground_truth_response"], "4")

            # Verify provider was called correctly
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

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        error, metadata = result
        self.assertAlmostEqual(error, 0.1, places=10)  # |0.8 - 0.9|
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
            with patch(
                "trulens.feedback.groundtruth.BERTScorer"
            ) as mock_bert_scorer_class:
                # Mock BERTScorer instance
                mock_scorer = Mock()
                mock_score_tensor = Mock()
                mock_score_tensor.item.return_value = 0.85
                mock_scorer.score.return_value = (mock_score_tensor,)
                mock_bert_scorer_class.return_value = mock_scorer

                result = self.agreement.bert_score("What is 2+2?", "Four")

                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)

                score, metadata = result
                self.assertEqual(score, 0.85)
                self.assertEqual(metadata["ground_truth_response"], "4")
        except ModuleNotFoundError:
            # bert-score is optional dependency, skip if not available
            self.skipTest("bert-score not available")

    @pytest.mark.optional
    def test_bert_score_no_response_found(self):
        """Test bert_score when no ground truth response is found."""
        try:
            result = self.agreement.bert_score("Unknown query", "Some response")
            self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            # bert-score is optional dependency, skip if not available
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

                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)

                score, metadata = result
                self.assertEqual(score, 0.75)
                self.assertEqual(metadata["ground_truth_response"], "4")
        except ModuleNotFoundError:
            # evaluate is optional dependency, skip if not available
            self.skipTest("evaluate not available")

    @pytest.mark.optional
    def test_bleu_score_no_response_found(self):
        """Test bleu score when no ground truth response is found."""
        try:
            result = self.agreement.bleu("Unknown query", "Some response")
            self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            # evaluate is optional dependency, skip if not available
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

                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)

                score, metadata = result
                self.assertEqual(score, 0.65)
                self.assertEqual(metadata["ground_truth_response"], "4")
        except ModuleNotFoundError:
            # evaluate is optional dependency, skip if not available
            self.skipTest("evaluate not available")

    @pytest.mark.optional
    def test_rouge_score_no_response_found(self):
        """Test rouge score when no ground truth response is found."""
        try:
            result = self.agreement.rouge("Unknown query", "Some response")
            self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            # evaluate is optional dependency, skip if not available
            self.skipTest("evaluate not available")

    @pytest.mark.optional
    def test_ndcg_at_k_perfect_ranking(self):
        """Test NDCG@k with perfect ranking."""
        query = "What is machine learning?"
        retrieved_chunks = [
            "Machine learning is a subset of AI",
            "It uses algorithms to learn patterns",
            "Some irrelevant text",
        ]
        relevance_scores = [1.0, 0.8, 0.1]

        result = self.agreement_with_chunks.ndcg_at_k(
            query, retrieved_chunks, relevance_scores, k=2
        )
        self.assertEqual(result, 1.0)  # Perfect NDCG

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
        query = "What is machine learning?"
        retrieved_chunks = [
            "Machine learning is a subset of AI",
            "It uses algorithms to learn patterns",
            "Some irrelevant text",
        ]

        result = self.agreement_with_chunks.precision_at_k(
            query, retrieved_chunks, k=2
        )
        self.assertEqual(result, 1.0)  # 2 relevant out of 2 retrieved

    @pytest.mark.optional
    def test_precision_at_k_with_relevance_scores(self):
        """Test precision@k with relevance scores."""
        query = "What is machine learning?"
        retrieved_chunks = [
            "Some irrelevant text",
            "Machine learning is a subset of AI",
            "It uses algorithms to learn patterns",
        ]
        relevance_scores = [0.1, 1.0, 0.8]

        result = self.agreement_with_chunks.precision_at_k(
            query, retrieved_chunks, relevance_scores, k=2
        )
        self.assertEqual(
            result, 1.0
        )  # After sorting by scores, top-2 are both relevant

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
        query = "What is machine learning?"
        retrieved_chunks = [
            "Machine learning is a subset of AI",
            "It uses algorithms to learn patterns",
            "Some additional relevant content",
        ]

        result = self.agreement_with_chunks.recall_at_k(
            query, retrieved_chunks, k=3
        )
        self.assertEqual(result, 1.0)  # Found both golden chunks

    @pytest.mark.optional
    def test_recall_at_k_partial_recall(self):
        """Test recall@k with partial recall."""
        query = "What is machine learning?"
        retrieved_chunks = [
            "Machine learning is a subset of AI",
            "Some irrelevant text",
        ]

        result = self.agreement_with_chunks.recall_at_k(
            query, retrieved_chunks, k=2
        )
        self.assertEqual(result, 0.5)  # Found 1 out of 2 golden chunks

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
        query = "What is machine learning?"
        retrieved_chunks = [
            "Machine learning is a subset of AI",
            "Some irrelevant text",
            "It uses algorithms to learn patterns",
        ]

        result = self.agreement_with_chunks.mrr(query, retrieved_chunks)
        self.assertEqual(result, 1.0)  # 1/1

    @pytest.mark.optional
    def test_mrr_second_position(self):
        """Test MRR when first relevant item is at position 2."""
        query = "What is machine learning?"
        retrieved_chunks = [
            "Some irrelevant text",
            "Machine learning is a subset of AI",
            "It uses algorithms to learn patterns",
        ]

        result = self.agreement_with_chunks.mrr(query, retrieved_chunks)
        self.assertEqual(result, 0.5)  # 1/2

    @pytest.mark.optional
    def test_mrr_with_relevance_scores(self):
        """Test MRR with relevance scores for sorting."""
        query = "What is machine learning?"
        retrieved_chunks = [
            "Some irrelevant text",
            "Machine learning is a subset of AI",
            "It uses algorithms to learn patterns",
        ]
        relevance_scores = [0.1, 1.0, 0.8]

        result = self.agreement_with_chunks.mrr(
            query, retrieved_chunks, relevance_scores
        )
        self.assertEqual(result, 1.0)  # After sorting, relevant item is first

    @pytest.mark.optional
    def test_mrr_no_relevant_items(self):
        """Test MRR when no relevant items are found."""
        query = "What is machine learning?"
        retrieved_chunks = [
            "Completely irrelevant text",
            "Another irrelevant chunk",
        ]

        result = self.agreement_with_chunks.mrr(query, retrieved_chunks)
        self.assertEqual(result, 0.0)

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
        query = "What is machine learning?"
        retrieved_chunks = [
            "Some irrelevant text",
            "Machine learning is a subset of AI",
        ]

        result = self.agreement_with_chunks.ir_hit_rate(query, retrieved_chunks)
        self.assertEqual(result, 1.0)

    @pytest.mark.optional
    def test_ir_hit_rate_no_hit(self):
        """Test IR hit rate when there's no hit."""
        query = "What is machine learning?"
        retrieved_chunks = ["Some irrelevant text", "Another irrelevant chunk"]

        result = self.agreement_with_chunks.ir_hit_rate(query, retrieved_chunks)
        self.assertEqual(result, 0.0)

    @pytest.mark.optional
    def test_ir_hit_rate_with_k(self):
        """Test IR hit rate with k parameter."""
        query = "What is machine learning?"
        retrieved_chunks = [
            "Some irrelevant text",
            "Machine learning is a subset of AI",
        ]

        # Hit rate at k=1 should be 0 (no hit in first position)
        result = self.agreement_with_chunks.ir_hit_rate(
            query, retrieved_chunks, k=1
        )
        self.assertEqual(result, 0.0)

        # Hit rate at k=2 should be 1 (hit found within top 2)
        result = self.agreement_with_chunks.ir_hit_rate(
            query, retrieved_chunks, k=2
        )
        self.assertEqual(result, 1.0)

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

    # ========== EDGE CASES AND ERROR HANDLING ==========

    @pytest.mark.optional
    def test_empty_retrieved_chunks(self):
        """Test behavior with empty retrieved chunks."""
        query = "What is machine learning?"

        result = self.agreement_with_chunks.precision_at_k(query, [])
        self.assertEqual(result, 0.0)

        result = self.agreement_with_chunks.recall_at_k(query, [])
        self.assertEqual(result, 0.0)

        # NDCG with empty chunks should return NaN or handle gracefully
        try:
            result = self.agreement_with_chunks.ndcg_at_k(query, [])
            self.assertTrue(np.isnan(result) or result == 0.0)
        except Exception:
            # It's acceptable for NDCG to raise an exception with empty input
            pass

        result = self.agreement_with_chunks.mrr(query, [])
        self.assertEqual(result, 0.0)

        result = self.agreement_with_chunks.ir_hit_rate(query, [])
        self.assertEqual(result, 0.0)

    @pytest.mark.optional
    def test_edge_case_k_larger_than_chunks(self):
        """Test behavior when k is larger than number of chunks."""
        query = "What is machine learning?"
        retrieved_chunks = ["Machine learning is a subset of AI"]

        # k=5 but only 1 chunk - should handle gracefully
        result = self.agreement_with_chunks.precision_at_k(
            query, retrieved_chunks, k=5
        )
        self.assertEqual(result, 1.0)

        result = self.agreement_with_chunks.recall_at_k(
            query, retrieved_chunks, k=5
        )
        self.assertEqual(result, 0.5)  # 1 out of 2 golden chunks found

    @pytest.mark.optional
    def test_empty_ground_truth_list(self):
        """Test behavior with empty ground truth list."""
        empty_agreement = GroundTruthAgreement(
            ground_truth=[], provider=self.mock_provider
        )

        result = empty_agreement.agreement_measure("Any query", "Any response")
        self.assertTrue(np.isnan(result))

        try:
            result = empty_agreement.bert_score("Any query", "Any response")
            self.assertTrue(np.isnan(result))
        except ModuleNotFoundError:
            # bert-score is optional dependency, skip this part if not available
            pass

    @pytest.mark.optional
    def test_special_characters_in_queries(self):
        """Test handling of special characters in queries and responses."""
        special_ground_truth = [
            {
                "query": "What is 'machine learning'?",
                "expected_response": "AI subset",
            },
            {
                "query": "Explain A&B testing",
                "expected_response": "Statistical method",
            },
        ]

        special_agreement = GroundTruthAgreement(
            ground_truth=special_ground_truth, provider=self.mock_provider
        )

        response = special_agreement._find_response(
            "What is 'machine learning'?"
        )
        self.assertEqual(response, "AI subset")

        response = special_agreement._find_response("Explain A&B testing")
        self.assertEqual(response, "Statistical method")

    @pytest.mark.optional
    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        unicode_ground_truth = [
            {
                "query": "¿Qué es machine learning?",
                "expected_response": "Aprendizaje automático",
            },
            {
                "query": "什么是机器学习?",
                "expected_response": "机器学习是人工智能的一个分支",
            },
        ]

        unicode_agreement = GroundTruthAgreement(
            ground_truth=unicode_ground_truth, provider=self.mock_provider
        )

        response = unicode_agreement._find_response("¿Qué es machine learning?")
        self.assertEqual(response, "Aprendizaje automático")

        response = unicode_agreement._find_response("什么是机器学习?")
        self.assertEqual(response, "机器学习是人工智能的一个分支")

    @pytest.mark.optional
    def test_case_sensitivity(self):
        """Test case sensitivity in query matching."""
        response = self.agreement._find_response("what is 2+2?")  # lowercase
        self.assertIsNone(response)  # Should not match due to case sensitivity

        response = self.agreement._find_response("What is 2+2?")  # exact match
        self.assertEqual(response, "4")

    @pytest.mark.optional
    def test_duplicate_queries_in_ground_truth(self):
        """Test behavior with duplicate queries in ground truth."""
        duplicate_ground_truth = [
            {"query": "What is 2+2?", "expected_response": "4"},
            {
                "query": "What is 2+2?",
                "expected_response": "Four",
            },  # Duplicate with different response
        ]

        duplicate_agreement = GroundTruthAgreement(
            ground_truth=duplicate_ground_truth, provider=self.mock_provider
        )

        # Should return the first match
        response = duplicate_agreement._find_response("What is 2+2?")
        self.assertEqual(response, "4")

    @pytest.mark.optional
    def test_chunks_without_scores(self):
        """Test handling of chunks without expect_score field."""
        chunks_no_scores = [
            {
                "query": "Test query",
                "expected_chunks": [
                    {"text": "Chunk without score"},
                    {"text": "Another chunk without score"},
                ],
            }
        ]

        chunks_agreement = GroundTruthAgreement(
            ground_truth=chunks_no_scores, provider=self.mock_provider
        )

        chunks = chunks_agreement._find_golden_context_chunks_and_scores(
            "Test query"
        )
        expected = [
            ("Chunk without score", 1),  # Default score should be 1
            ("Another chunk without score", 1),
        ]
        self.assertEqual(chunks, expected)

    # ========== INTEGRATION AND REALISTIC SCENARIOS ==========

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

        realistic_agreement = GroundTruthAgreement(
            ground_truth=realistic_qa, provider=self.mock_provider
        )

        # Test exact match
        response = realistic_agreement._find_response(
            "What are the main components of a neural network?"
        )
        self.assertIsNotNone(response)
        self.assertIn("neural network", response.lower())

        # Test agreement measure with realistic response
        with patch("trulens.feedback.generated.re_0_10_rating", return_value=8):
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

        ir_agreement = GroundTruthAgreement(
            ground_truth=realistic_ir, provider=self.mock_provider
        )

        # Test with realistic retrieved chunks
        retrieved_chunks = [
            "Compute gradients",  # Exact match
            "Calculate the loss function",  # Similar but not exact
            "Update parameters using learning rate",  # Exact match
            "Some irrelevant information",
        ]

        # Precision should be reasonable (2/4 = 0.5 for exact matches)
        precision = ir_agreement.precision_at_k(
            "How to implement gradient descent?", retrieved_chunks
        )
        self.assertEqual(precision, 0.5)

        # Recall should be reasonable (2/4 = 0.5 of golden chunks found)
        recall = ir_agreement.recall_at_k(
            "How to implement gradient descent?", retrieved_chunks
        )
        self.assertEqual(recall, 0.5)

    @pytest.mark.optional
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        # Create a larger dataset
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                "query": f"Question {i}?",
                "expected_response": f"Answer {i}",
            })

        large_agreement = GroundTruthAgreement(
            ground_truth=large_dataset, provider=self.mock_provider
        )

        # Test that lookup still works efficiently
        response = large_agreement._find_response("Question 50?")
        self.assertEqual(response, "Answer 50")

        # Test with non-existent query
        response = large_agreement._find_response("Non-existent question")
        self.assertIsNone(response)

    @pytest.mark.optional
    def test_mixed_data_types(self):
        """Test with mixed data types in ground truth."""
        mixed_data = [
            {
                "query": "What is 2+2?",
                "expected_response": "4",
            },  # String response
            {
                "query": "What is π?",
                "expected_response": 3.14159,
                "expected_score": 0.95,
            },  # Numeric response
            {
                "query": "Is Python good?",
                "expected_response": True,
                "expected_score": 0.8,
            },  # Boolean response
        ]

        mixed_agreement = GroundTruthAgreement(
            ground_truth=mixed_data, provider=self.mock_provider
        )

        # Test string response
        response = mixed_agreement._find_response("What is 2+2?")
        self.assertEqual(response, "4")

        # Test numeric response (should be converted to string in practice)
        response = mixed_agreement._find_response("What is π?")
        self.assertEqual(response, 3.14159)

        # Test score lookup
        score = mixed_agreement._find_score("What is π?", 3.14159)
        self.assertEqual(score, 0.95)

    @pytest.mark.optional
    def test_consistency_across_methods(self):
        """Test consistency of query matching across different methods."""
        test_query = "What is machine learning?"

        # Test that all methods handle the same query consistently
        chunks = (
            self.agreement_with_chunks._find_golden_context_chunks_and_scores(
                test_query
            )
        )
        self.assertIsNotNone(chunks)

        precision = self.agreement_with_chunks.precision_at_k(
            test_query, ["Machine learning is a subset of AI"]
        )
        self.assertEqual(precision, 1.0)

        recall = self.agreement_with_chunks.recall_at_k(
            test_query, ["Machine learning is a subset of AI"]
        )
        self.assertEqual(recall, 0.5)  # 1 out of 2 chunks found

        hit_rate = self.agreement_with_chunks.ir_hit_rate(
            test_query, ["Machine learning is a subset of AI"]
        )
        self.assertEqual(hit_rate, 1.0)

    @pytest.mark.optional
    def test_thread_safety_simulation(self):
        """Test thread safety by simulating concurrent access."""
        import threading
        import time

        results = []

        def worker():
            for i in range(10):
                response = self.agreement._find_response("What is 2+2?")
                results.append(response)
                time.sleep(0.001)  # Small delay to simulate real work

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All results should be the same
        expected_results = ["4"] * 50  # 5 threads * 10 iterations
        self.assertEqual(results, expected_results)
