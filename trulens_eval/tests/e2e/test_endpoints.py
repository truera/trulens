"""Tests endpoints and specifically cost tracking.

These tests make use of potentially non-free apis and require various secrets
configured. See `setUp` below.
"""

import os
from pprint import PrettyPrinter
from unittest import main
from unittest import skip
from unittest import TestCase

from tests.unit.test import optional_test

from trulens_eval.feedback.provider.endpoint import Endpoint
from trulens_eval.keys import check_keys

pp = PrettyPrinter()


class TestEndpoints(TestCase):
    """Tests for cost tracking of endpoints."""

    def setUp(self):
        check_keys(
            # for non-azure openai tests
            "OPENAI_API_KEY",

            # for huggingface tests
            "HUGGINGFACE_API_KEY",

            # for bedrock tests
            "AWS_REGION_NAME",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",

            # for azure openai tests
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",

            # for snowflake cortex
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER",
            "SNOWFLAKE_USER_PASSWORD"
        )

    def _test_llm_provider_endpoint(self, provider, with_cost: bool = True):
        """Cost checks for endpoints whose providers implement LLMProvider."""

        _, cost = Endpoint.track_all_costs_tally(
            provider.sentiment, text="This rocks!"
        )

        self.assertEqual(cost.n_requests, 1, "Expected exactly one request.")
        self.assertEqual(
            cost.n_successful_requests, 1,
            "Expected exactly one successful request."
        )
        self.assertEqual(
            cost.n_classes, 0, "Expected zero classes for LLM-based endpoints."
        )
        self.assertEqual(
            cost.n_stream_chunks, 0,
            "Expected zero chunks when not using streaming mode."
        )
        self.assertGreater(cost.n_tokens, 0, "Expected non-zero tokens.")
        self.assertGreater(
            cost.n_prompt_tokens, 0, "Expected non-zero prompt tokens."
        )
        self.assertGreater(
            cost.n_completion_tokens, 0.0,
            "Expected non-zero completion tokens."
        )

        if with_cost:
            self.assertGreater(cost.cost, 0.0, "Expected non-zero cost.")

    @skip("This test needs to be updated.")
    @optional_test
    def test_hugs(self):
        """Check that cost tracking works for the huggingface endpoint."""

        from trulens_eval.feedback.provider import Huggingface

        hugs = Huggingface()

        _, cost = Endpoint.track_all_costs_tally(
            hugs.positive_sentiment, text="This rocks!"
        )

        self.assertEqual(cost.n_requests, 1, "Expected exactly one request.")
        self.assertEqual(
            cost.n_successful_requests, 1,
            "Expected exactly one successful request."
        )
        self.assertEqual(
            cost.n_classes, 3,
            "Expected exactly three classes for sentiment classification."
        )
        self.assertEqual(
            cost.n_stream_chunks, 0,
            "Expected zero chunks for classification endpoints."
        )
        self.assertEqual(cost.n_tokens, 0, "Expected zero tokens.")
        self.assertEqual(
            cost.n_prompt_tokens, 0, "Expected zero prompt tokens."
        )
        self.assertEqual(
            cost.n_completion_tokens, 0.0, "Expected zero completion tokens."
        )

        self.assertEqual(
            cost.cost, 0.0, "Expected zero cost for huggingface endpoint."
        )

    @skip("This test needs to be updated.")
    @optional_test
    def test_openai(self):
        """Check that cost tracking works for openai models."""

        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["OPENAI_API_TYPE"] = "openai"

        from trulens_eval.feedback.provider.openai import OpenAI

        provider = OpenAI(model_engine=OpenAI.DEFAULT_MODEL_ENGINE)

        self._test_llm_provider_endpoint(provider)

    @skip("This test needs to be updated.")
    @optional_test
    def test_litellm_openai(self):
        """Check that cost tracking works for openai models through litellm."""

        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["OPENAI_API_TYPE"] = "openai"

        from trulens_eval.feedback.provider import LiteLLM
        from trulens_eval.feedback.provider.openai import OpenAI

        # Have to delete litellm endpoint singleton as it may have been created
        # with the wrong underlying litellm provider in a prior test.
        Endpoint.delete_singleton_by_name("litellm")

        provider = LiteLLM(f"openai/{OpenAI.DEFAULT_MODEL_ENGINE}")

        self._test_llm_provider_endpoint(provider)

    @skip("This test needs to be updated.")
    @optional_test
    def test_openai_azure(self):
        """Check that cost tracking works for openai azure models."""

        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["OPENAI_API_TYPE"] = "azure"

        from trulens_eval.feedback.provider.openai import AzureOpenAI

        provider = AzureOpenAI(
            model_engine=AzureOpenAI.DEFAULT_MODEL_ENGINE,
            deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']
        )

        self._test_llm_provider_endpoint(provider)

    @skip("This test needs to be updated.")
    @optional_test
    def test_litellm_openai_azure(self):
        """Check that cost tracking works for openai models through litellm."""

        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["OPENAI_API_TYPE"] = "azure"

        # Have to delete litellm endpoint singleton as it may have been created
        # with the wrong underlying litellm provider in a prior test.
        Endpoint.delete_singleton_by_name("litellm")

        from trulens_eval.feedback.provider import LiteLLM

        provider = LiteLLM(
            f"azure/{os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']}",
            completion_kwargs=dict(
                api_base=os.environ['AZURE_OPENAI_ENDPOINT']
            )
        )

        self._test_llm_provider_endpoint(provider)

    @skip("This test needs to be updated.")
    @optional_test
    def test_bedrock(self):
        """Check that cost tracking works for bedrock models."""

        from trulens_eval.feedback.provider.bedrock import Bedrock

        provider = Bedrock(model_id=Bedrock.DEFAULT_MODEL_ID)

        # We don't have USD cost tracking for bedrock or anything beyond openai.
        self._test_llm_provider_endpoint(provider, with_cost=False)

    @skip("This test needs to be updated.")
    @optional_test
    def test_litellm_bedrock(self):
        """Check that cost tracking works for bedrock models through litellm."""

        from trulens_eval.feedback.provider import LiteLLM
        from trulens_eval.feedback.provider.bedrock import Bedrock

        # Have to delete litellm endpoint singleton as it may have been created
        # with the wrong underlying litellm provider in a prior test.
        Endpoint.delete_singleton_by_name("litellm")

        provider = LiteLLM(f"bedrock/{Bedrock.DEFAULT_MODEL_ID}")

        # Litellm comes with cost tracking for bedrock though it may be inaccurate.
        self._test_llm_provider_endpoint(provider)

    @optional_test
    def test_cortex(self):
        """Check that cost (token) tracking works for Cortex LLM Functions"""
        from trulens_eval.feedback.provider.cortex import Cortex

        provider = Cortex(model_engine="snowflake-arctic")

        self._test_llm_provider_endpoint(provider)


if __name__ == '__main__':
    main()
