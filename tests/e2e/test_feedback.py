"""
Tests for feedback functions.
"""

import os

import pytest
from trulens.core.feedback import feedback as core_feedback
from trulens.providers.cortex import Cortex

from tests.util.snowflake_test_case import SnowflakeTestCase


@pytest.mark.snowflake
class TestFeedback(SnowflakeTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._orig_TRULENS_OTEL_TRACING = os.getenv("TRULENS_OTEL_TRACING")
        os.environ["TRULENS_OTEL_TRACING"] = "0"
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._orig_TRULENS_OTEL_TRACING is not None:
            os.environ["TRULENS_OTEL_TRACING"] = cls._orig_TRULENS_OTEL_TRACING
        else:
            del os.environ["TRULENS_OTEL_TRACING"]
        return super().tearDownClass()

    def test_groundedness_with_nuggetization(self) -> None:
        """Test groundedness with nuggetization splitter and print nuggets."""
        # Create Cortex provider using the session from SnowflakeTestCase
        # Using a more capable model for better nugget extraction
        provider = Cortex(
            snowpark_session=self._snowpark_session,
            model_engine="llama3.1-70b",
        )

        # Sample source and statement for groundedness evaluation
        source = """
        The University of Washington (UW) is a public research university
        in Seattle, Washington. Founded in 1861, it is one of the oldest
        universities on the West Coast. The university has over 46,000
        students and is known for its research programs. UW has strong
        connections to major technology companies like Microsoft and Amazon,
        which are headquartered in the Seattle area.
        """

        statement = """
        The University of Washington is a public research university
        located in Seattle. It was established in 1861, making it one
        of the oldest universities on the West Coast. The university
        has close ties to major tech companies in the Seattle area.
        """
        statement = "- **4. Proof**: 3 opportunities totaling $987,132.81 USD across Company ABC and Company Canada Frontline"
        statement = "- **5. Agreement**: 1 opportunity totaling $1,1234.25 USD (Company A Inc.), "

        # Create groundedness config with nuggetize splitter
        groundedness_configs = core_feedback.GroundednessConfigs(
            use_sent_tokenize=False,
            filter_trivial_statements=False,
            splitter="nuggetize",
        )

        # Call groundedness_measure_with_cot_reasons with nuggetize config
        score, metadata = provider.groundedness_measure_with_cot_reasons(
            source=source,
            statement=statement,
            groundedness_configs=groundedness_configs,
        )

        # Print the results
        print(f"\n{'=' * 60}")
        print("Groundedness with Nuggetization Results")
        print(f"{'=' * 60}")
        print(f"Overall Score: {score}")
        print(f"Total Nuggets: {metadata.get('total_nuggets', 0)}")
        print(f"Method: {metadata.get('method', 'unknown')}")
        print(f"\n{'=' * 60}")
        print("Nugget Evaluations:")
        print(f"{'=' * 60}")

        nugget_evaluations = metadata.get("nugget_evaluations", [])
        for i, eval_dict in enumerate(nugget_evaluations):
            print(f"\nNugget {i + 1}:")
            print(f"  Text: {eval_dict.get('nugget', 'N/A')}")
            print(f"  Importance: {eval_dict.get('importance', 'N/A')}")
            print(f"  Score: {eval_dict.get('score', 'N/A')}")
            print(f"  Reason: {eval_dict.get('reason', 'N/A')}")

        print(f"\n{'=' * 60}")

        # Basic assertions to verify the result
        self.assertIsInstance(score, float)
        self.assertEqual(metadata.get("method"), "nuggetized")
        self.assertIn("nugget_evaluations", metadata)
        self.assertIn("total_nuggets", metadata)

    def test_groundedness_with_sent_tokenize(self) -> None:
        """Test groundedness with sent_tokenize splitter and print sentences."""
        # Create Cortex provider using the session from SnowflakeTestCase
        provider = Cortex(
            snowpark_session=self._snowpark_session,
            model_engine="llama3.1-70b",
        )

        # Sample source and statement for groundedness evaluation
        source = """
        The University of Washington (UW) is a public research university
        in Seattle, Washington. Founded in 1861, it is one of the oldest
        universities on the West Coast. The university has over 46,000
        students and is known for its research programs. UW has strong
        connections to major technology companies like Microsoft and Amazon,
        which are headquartered in the Seattle area.
        """

        statement = """
        The University of Washington is a public research university
        located in Seattle. It was established in 1861, making it one
        of the oldest universities on the West Coast. The university
        has close ties to major tech companies in the Seattle area.
        """
        statement = "- **4. Proof**: 3 opportunities totaling $987,132.81 USD across Company ABC and Company Canada Frontline"
        statement = "- **5. Agreement**: 1 opportunity totaling $1,1234.25 USD (Company A Inc.), "

        # Create groundedness config with sent_tokenize splitter
        groundedness_configs = core_feedback.GroundednessConfigs(
            use_sent_tokenize=True,
            filter_trivial_statements=False,
            splitter="sent_tokenize",
        )

        # Call groundedness_measure_with_cot_reasons with sent_tokenize config
        score, metadata = provider.groundedness_measure_with_cot_reasons(
            source=source,
            statement=statement,
            groundedness_configs=groundedness_configs,
        )

        # Print the results
        print(f"\n{'=' * 60}")
        print("Groundedness with Sent Tokenize Results")
        print(f"{'=' * 60}")
        print(f"Overall Score: {score}")
        print(f"\n{'=' * 60}")
        print("Sentence Evaluations:")
        print(f"{'=' * 60}")

        reasons = metadata.get("reasons", [])
        for i, reason in enumerate(reasons):
            print(f"\nSentence {i + 1}:")
            print(f"  Criteria: {reason.get('criteria', 'N/A')}")
            print(f"  Evidence: {reason.get('supporting_evidence', 'N/A')}")
            print(f"  Score: {reason.get('score', 'N/A')}")

        print(f"\n{'=' * 60}")

        # Basic assertions to verify the result
        self.assertIsInstance(score, float)
        self.assertIn("reasons", metadata)

    def test_groundedness_with_llm_splitter(self) -> None:
        """Test groundedness with LLM splitter and print sentences."""
        # Create Cortex provider using the session from SnowflakeTestCase
        provider = Cortex(
            snowpark_session=self._snowpark_session,
            model_engine="llama3.1-70b",
        )

        # Sample source and statement for groundedness evaluation
        source = """
        The University of Washington (UW) is a public research university
        in Seattle, Washington. Founded in 1861, it is one of the oldest
        universities on the West Coast. The university has over 46,000
        students and is known for its research programs. UW has strong
        connections to major technology companies like Microsoft and Amazon,
        which are headquartered in the Seattle area.
        """

        statement = """
        The University of Washington is a public research university
        located in Seattle. It was established in 1861, making it one
        of the oldest universities on the West Coast. The university
        has close ties to major tech companies in the Seattle area.
        """
        statement = "- **4. Proof**: 3 opportunities totaling $987,132.81 USD across Company ABC and Company Canada Frontline"
        statement = "- **5. Agreement**: 1 opportunity totaling $1,1234.25 USD (Company A Inc.), "

        # Create groundedness config with llm splitter
        groundedness_configs = core_feedback.GroundednessConfigs(
            use_sent_tokenize=False,
            filter_trivial_statements=False,
            splitter="llm",
        )

        # Call groundedness_measure_with_cot_reasons with llm config
        score, metadata = provider.groundedness_measure_with_cot_reasons(
            source=source,
            statement=statement,
            groundedness_configs=groundedness_configs,
        )

        # Print the results
        print(f"\n{'=' * 60}")
        print("Groundedness with LLM Splitter Results")
        print(f"{'=' * 60}")
        print(f"Overall Score: {score}")
        print(f"\n{'=' * 60}")
        print("Sentence Evaluations:")
        print(f"{'=' * 60}")

        reasons = metadata.get("reasons", [])
        for i, reason in enumerate(reasons):
            print(f"\nSentence {i + 1}:")
            print(f"  Criteria: {reason.get('criteria', 'N/A')}")
            print(f"  Evidence: {reason.get('supporting_evidence', 'N/A')}")
            print(f"  Score: {reason.get('score', 'N/A')}")

        print(f"\n{'=' * 60}")

        # Basic assertions to verify the result
        self.assertIsInstance(score, float)
        self.assertIn("reasons", metadata)
