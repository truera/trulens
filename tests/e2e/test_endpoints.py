"""Tests endpoints.

These tests make use of potentially non-free apis and require
various secrets configured. See `setUp` below.
"""

import os
from pprint import PrettyPrinter
from unittest import skip

import pytest
from snowflake.snowpark import Session
from trulens.core import experimental as core_experimental
from trulens.core import session as core_session
from trulens.core.feedback import endpoint as core_endpoint
from trulens.core.utils import keys as key_utils

from tests import test as test_utils

pp = PrettyPrinter()


class TestEndpoints(test_utils.TruTestCase):
    """Tests for cost tracking of endpoints."""

    @classmethod
    def setUpClass(cls):
        session = core_session.TruSession()

        if cls.env_true(test_utils.USE_OTEL_TRACING):
            session.experimental_enable_feature(
                core_experimental.Feature.OTEL_TRACING
            )

    def setUp(self):
        key_utils.check_keys(
            # for non-azure openai tests
            "OPENAI_API_KEY",
            # for huggingface tests
            # "HUGGINGFACE_API_KEY",
            # for bedrock tests # no current keys available for bedrock
            # "AWS_REGION_NAME",
            # "AWS_ACCESS_KEY_ID",
            # "AWS_SECRET_ACCESS_KEY",
            # "AWS_SESSION_TOKEN",
            # for azure openai tests
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "OPENAI_API_VERSION",
            # for snowflake cortex
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER",
            "SNOWFLAKE_USER_PASSWORD",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
            "SNOWFLAKE_WAREHOUSE",
        )

    def _test_hugs_provider_endpoint(self, provider, with_cost: bool = True):
        """Check that cost tracking works for the huggingface endpoints."""

        _, cost_tally = core_endpoint.Endpoint.track_all_costs_tally(
            provider.positive_sentiment, text="This rocks!"
        )
        cost = cost_tally()

        with self.subTest("n_requests"):
            self.assertEqual(
                cost.n_requests, 1, "Expected exactly one request."
            )

        with self.subTest("n_successful_requests"):
            self.assertEqual(
                cost.n_successful_requests,
                1,
                "Expected exactly one successful request.",
            )

        with self.subTest("n_classes"):
            self.assertEqual(
                cost.n_classes,
                3,
                "Expected exactly three classes for sentiment classification.",
            )

        with self.subTest("n_stream_chunks"):
            self.assertEqual(
                cost.n_stream_chunks,
                0,
                "Expected zero chunks for classification endpoints.",
            )

        with self.subTest("n_tokens"):
            self.assertEqual(cost.n_tokens, 0, "Expected zero tokens.")

        with self.subTest("n_prompt_tokens"):
            self.assertEqual(
                cost.n_prompt_tokens, 0, "Expected zero prompt tokens."
            )

        with self.subTest("n_completion_tokens"):
            self.assertEqual(
                cost.n_completion_tokens,
                0.0,
                "Expected zero completion tokens.",
            )

        if with_cost:
            with self.subTest("cost"):
                self.assertEqual(
                    cost.cost,
                    0.0,
                    "Expected zero cost for huggingface endpoint.",
                )

    def _test_llm_provider_endpoint(self, provider, with_cost: bool = True):
        """Cost checks for endpoints whose providers implement LLMProvider."""

        _, cost_tally = core_endpoint.Endpoint.track_all_costs_tally(
            provider.sentiment, text="This rocks!"
        )
        cost = cost_tally()

        with self.subTest("n_requests"):
            self.assertEqual(
                cost.n_requests, 1, "Expected exactly one request."
            )

        with self.subTest("n_successful_requests"):
            self.assertEqual(
                cost.n_successful_requests,
                1,
                "Expected exactly one successful request.",
            )

        with self.subTest("n_classes"):
            self.assertEqual(
                cost.n_classes,
                0,
                "Expected zero classes for LLM-based endpoints.",
            )

        with self.subTest("n_stream_chunks"):
            self.assertEqual(
                cost.n_stream_chunks,
                0,
                "Expected zero chunks when not using streaming mode.",
            )

        with self.subTest("n_tokens"):
            self.assertGreater(cost.n_tokens, 0, "Expected non-zero tokens.")

        with self.subTest("n_prompt_tokens"):
            self.assertGreater(
                cost.n_prompt_tokens, 0, "Expected non-zero prompt tokens."
            )

        with self.subTest("n_completion_tokens"):
            self.assertGreater(
                cost.n_completion_tokens,
                0.0,
                "Expected non-zero completion tokens.",
            )

        if with_cost:
            with self.subTest("cost"):
                self.assertGreater(cost.cost, 0.0, "Expected non-zero cost.")

        if (
            str(type(provider))
            == "<class 'trulens.providers.cortex.provider.Cortex'>"
        ):
            with self.subTest("n_cortex_guardrails_tokens"):
                self.assertGreater(
                    cost.n_cortex_guardrails_tokens,
                    0.0,
                    "Expected non-zero cortex guardrails tokens.",
                )

            with self.subTest("cost_currency"):
                self.assertEqual(
                    cost.cost_currency,
                    "Snowflake credits",
                    "Expected cost currency to be Snowflake credits.",
                )

    @pytest.mark.optional
    def test_dummy_hugs(self):
        """Check that cost tracking works for the dummy huggingface provider."""

        from trulens.providers.huggingface.provider import Dummy

        self._test_hugs_provider_endpoint(Dummy())

    def test_dummy_llm(self):
        """Check that cost tracking works for dummy llm provider."""

        from trulens.feedback.dummy.provider import DummyProvider

        self._test_llm_provider_endpoint(DummyProvider())

    @pytest.mark.optional
    def test_hugs(self):
        """Check that cost tracking works for the huggingface endpoint."""

        from trulens.providers.huggingface import Huggingface

        self._test_hugs_provider_endpoint(Huggingface())

    @pytest.mark.optional
    def test_openai(self):
        """Check that cost tracking works for openai models."""

        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["OPENAI_API_TYPE"] = "openai"

        from trulens.providers.openai import OpenAI

        provider = OpenAI(model_engine=OpenAI.DEFAULT_MODEL_ENGINE)

        self._test_llm_provider_endpoint(provider)

    @pytest.mark.optional
    def test_litellm_openai(self):
        """Check that cost tracking works for openai models through litellm."""

        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["OPENAI_API_TYPE"] = "openai"

        from trulens.providers.litellm import LiteLLM
        from trulens.providers.litellm.endpoint import LiteLLMEndpoint
        from trulens.providers.openai import OpenAI
        from trulens.providers.openai.endpoint import OpenAIEndpoint

        # Have to delete litellm endpoint singleton as it may have been created
        # with the wrong underlying litellm provider in a prior test.
        LiteLLMEndpoint.delete_instances()
        OpenAIEndpoint.delete_instances()

        provider = LiteLLM(f"openai/{OpenAI.DEFAULT_MODEL_ENGINE}")

        self._test_llm_provider_endpoint(provider)

    @pytest.mark.optional
    def test_openai_azure(self):
        """Check that cost tracking works for openai azure models."""

        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["OPENAI_API_TYPE"] = "azure"

        from trulens.providers.openai import AzureOpenAI
        from trulens.providers.openai.endpoint import OpenAIEndpoint

        OpenAIEndpoint.delete_instances()

        provider = AzureOpenAI(
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["OPENAI_API_VERSION"],
        )

        self._test_llm_provider_endpoint(
            provider, with_cost=False
        )  # no cost tracking for azure openai when using Snowflake's deployment

    @pytest.mark.optional
    def test_litellm_openai_azure(self):
        """Check that cost tracking works for openai models through litellm."""

        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["AZURE_API_BASE"] = os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )  # this is required to be set explicitly for some weird reason alongside the api_base in completion_kwargs

        # Have to delete litellm endpoint singleton as it may have been created
        # with the wrong underlying litellm provider in a prior test.
        from trulens.providers.litellm.endpoint import LiteLLMEndpoint

        LiteLLMEndpoint.delete_instances()

        from trulens.providers.litellm import LiteLLM

        provider = LiteLLM(
            f"azure/{os.environ['AZURE_OPENAI_DEPLOYMENT']}",
            completion_kwargs=dict(
                api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
            ),
        )

        self._test_llm_provider_endpoint(provider)

    @skip("No keys available.")
    @pytest.mark.optional
    def test_bedrock(self):
        """Check that cost tracking works for bedrock models."""

        from trulens.providers.bedrock import Bedrock

        provider = Bedrock(model_id=Bedrock.DEFAULT_MODEL_ID)

        # We don't have USD cost tracking for bedrock or anything beyond openai.
        self._test_llm_provider_endpoint(provider, with_cost=False)

    @skip("No keys available.")
    @pytest.mark.optional
    def test_litellm_bedrock(self):
        """Check that cost tracking works for bedrock models through litellm."""

        from trulens.providers.bedrock import Bedrock
        from trulens.providers.litellm import LiteLLM

        # Have to delete litellm endpoint singleton as it may have been created
        # with the wrong underlying litellm provider in a prior test.
        from trulens.providers.litellm.endpoint import LiteLLMEndpoint

        LiteLLMEndpoint.delete_instances()

        provider = LiteLLM(f"bedrock/{Bedrock.DEFAULT_MODEL_ID}")

        # Litellm comes with cost tracking for bedrock though it may be inaccurate.
        self._test_llm_provider_endpoint(provider)

    @pytest.mark.optional
    def test_cortex(self):
        """Check that cost (token) tracking works for Cortex LLM Functions"""
        from trulens.providers.cortex import Cortex

        snowflake_connection_parameters = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ["SNOWFLAKE_DATABASE"],
            "schema": os.environ["SNOWFLAKE_SCHEMA"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        }

        snowpark_session = Session.builder.configs(
            snowflake_connection_parameters
        ).create()
        provider = Cortex(
            snowpark_session=snowpark_session,
            model_engine="snowflake-arctic",
        )

        self._test_llm_provider_endpoint(provider)
