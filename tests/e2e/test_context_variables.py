"""
Tests for context variable issues.
"""

import pytest
from trulens.apps.app import TruApp
from trulens.apps.app import instrument

from tests.util.snowflake_test_case import SnowflakeTestCase


class TestContextVariables(SnowflakeTestCase):
    @pytest.mark.optional
    def test_endpoint_contextvar_always_cleaned(self):
        class FailingRAG:
            @instrument
            def retrieve(self, query: str) -> list:
                return ["A", "B", "C"]

            @instrument
            def generate_completion(self, query: str, context_str: list) -> str:
                raise ValueError()

            @instrument
            def query(self, query: str) -> str:
                context_str = self.retrieve(query=query)
                completion = self.generate_completion(
                    query=query, context_str=context_str
                )
                return completion

        # Set up trulens.
        self.get_session("test_endpoint_contextvar_always_cleaned")
        rag = FailingRAG()
        tru_rag = TruApp(
            rag,
            app_name="FailingRAG",
            app_version="base",
        )

        with tru_rag:
            self.assertRaises(ValueError, rag.query, "X")
        # During app invocation, the endpoint context variable is set to track
        # costs, but because in this test it fails prematurely, we must make
        # sure the context variable is cleaned up properly. When it's set,
        # the snowflake.snowpark.Session.sql function is handled differently
        # in such a way that the following call will fail.
        # TODO: find a better way to check if the context variable is cleaned.
        self._snowpark_session.sql("SELECT 1").collect()
