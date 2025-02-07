"""
Tests for context variable issues.
"""

import os
import unittest

import pytest
from snowflake.snowpark import Session
from trulens.apps.custom import TruCustomApp
from trulens.apps.custom import instrument
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.core import TruSession


class TestContextVariables(unittest.TestCase):
    def setUp(self):
        connection_parameters = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ["SNOWFLAKE_DATABASE"],
            "role": os.environ["SNOWFLAKE_ROLE"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
            "schema": "TestContextVariables",
        }
        self._snowpark_session = Session.builder.configs(
            connection_parameters
        ).create()
        connector = SnowflakeConnector(
            **connection_parameters,
            init_server_side=True,
            init_server_side_with_staged_packages=True,
        )
        self._session = TruSession(connector=connector)

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
        rag = FailingRAG()
        tru_rag = TruCustomApp(
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


if __name__ == "__main__":
    unittest.main()
