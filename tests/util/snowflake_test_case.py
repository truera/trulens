"""
Test class to use for Snowflake testing.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional
import uuid

from snowflake.snowpark import Session
from snowflake.snowpark.row import Row
from trulens.connectors import snowflake as snowflake_connector
from trulens.core import session as core_session
from trulens.core.session import TruSession
from trulens.providers.cortex.provider import Cortex

from tests.test import TruTestCase


class SnowflakeTestCase(TruTestCase):
    def setUp(self):
        super().setUp()
        self._logger = logging.getLogger(__name__)
        self._database = os.environ["SNOWFLAKE_DATABASE"]
        self._snowflake_connection_parameters: Dict[str, str] = {
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "password": os.environ["SNOWFLAKE_USER_PASSWORD"],
            "database": os.environ["SNOWFLAKE_DATABASE"],
            "role": os.environ["SNOWFLAKE_ROLE"],
            "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        }
        self._snowpark_session = Session.builder.configs(
            self._snowflake_connection_parameters
        ).create()
        self._snowflake_schemas_to_delete = set()
        Cortex.DEFAULT_SNOWPARK_SESSION = self._snowpark_session

    def tearDown(self):
        # [HACK!] Clean up any instances of `TruSession` so tests don't interfere with each other.
        for key in [
            curr
            for curr in core_session.TruSession._singleton_instances
            if curr[0] == "trulens.core.session.TruSession"
        ]:
            del core_session.TruSession._singleton_instances[key]
        # Clean up any Snowflake schemas.
        schemas_not_deleted = []
        for curr in self._snowflake_schemas_to_delete:
            try:
                self.run_query(
                    f"DROP SCHEMA {self._snowflake_connection_parameters['database']}.{curr}"
                )
            except Exception:
                schemas_not_deleted.append(curr)
                self._logger.error(f"Failed to clean up schema {curr}!")
        # Check if any artifacts weren't deleted.
        if schemas_not_deleted:
            error_msg = "Failed to clean up the following schemas:\n"
            error_msg += "\n".join(schemas_not_deleted)
            raise ValueError(error_msg)
        # Close session.
        self._snowpark_session.close()
        super().tearDown()

    def list_schemas(self):
        res = self.run_query(
            f"SHOW SCHEMAS IN DATABASE {self._snowflake_connection_parameters['database']}"
        )
        return [curr["name"] for curr in res]

    def get_snowpark_session_with_schema(self, schema: str) -> Session:
        snowflake_connection_parameters = (
            self._snowflake_connection_parameters.copy()
        )
        snowflake_connection_parameters["schema"] = schema
        return Session.builder.configs(snowflake_connection_parameters).create()

    def get_session(
        self,
        app_base_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        schema_already_exists: bool = False,
        connect_via_snowpark_session: bool = True,
        init_server_side: bool = True,
        init_server_side_with_staged_packages: bool = True,
        use_account_event_table: bool = True,
    ) -> core_session.TruSession:
        if bool(app_base_name) == bool(schema_name):
            raise ValueError(
                "Exactly one of `app_base_name` and `schema_name` must be supplied!"
            )
        if app_base_name:
            app_name = app_base_name
            app_name += "__"
            app_name += str(uuid.uuid4()).replace("-", "_")
            self._schema = app_name
        else:
            self._schema = schema_name
        self._schema = self._schema.upper()
        if not schema_already_exists:
            self.assertNotIn(self._schema, self.list_schemas())
            self._snowflake_schemas_to_delete.add(self._schema)
        if not connect_via_snowpark_session:
            connector = snowflake_connector.SnowflakeConnector(
                schema=self._schema,
                **self._snowflake_connection_parameters,
                init_server_side=init_server_side,
                init_server_side_with_staged_packages=init_server_side_with_staged_packages,
                use_account_event_table=use_account_event_table,
            )
        else:
            if not schema_already_exists:
                self.create_and_use_schema(self._schema)
            connector = snowflake_connector.SnowflakeConnector(
                snowpark_session=self._snowpark_session,
                init_server_side=init_server_side,
                init_server_side_with_staged_packages=init_server_side_with_staged_packages,
                use_account_event_table=use_account_event_table,
            )
        session = core_session.TruSession(connector=connector)
        self.assertIn(self._schema, self.list_schemas())
        return session

    def run_query(
        self, q: str, bindings: Optional[List[Any]] = None
    ) -> List[Row]:
        return self._snowpark_session.sql(q, bindings).collect()

    def create_and_use_schema(
        self,
        schema_name: str,
        append_uuid: bool = False,
        delete_schema_on_cleanup: bool = True,
    ) -> str:
        schema_name = schema_name.upper()
        if append_uuid:
            schema_name = (
                f"{schema_name}__{str(uuid.uuid4()).replace('-', '_')}"
            )
        self._schema = schema_name
        self.run_query(
            "CREATE SCHEMA IF NOT EXISTS IDENTIFIER(?)", [schema_name]
        )
        if delete_schema_on_cleanup:
            self._snowflake_schemas_to_delete.add(schema_name)
        self._snowpark_session.use_schema(schema_name)
        return schema_name

    def _validate_num_spans_for_app(
        self,
        app_name: str,
        num_expected_spans: int,
        app_type: str = "EXTERNAL AGENT",
    ) -> List[Row]:
        # Flush exporter and wait for data to be made to stage.
        TruSession().force_flush()
        # Check that there are no other tables in the schema.
        self.assertListEqual(self.run_query("SHOW TABLES"), [])
        # Check that the data is in the event table.
        return self._wait_for_num_results(
            """
            SELECT
                *
            FROM
                table(snowflake.local.GET_AI_OBSERVABILITY_EVENTS(
                    ?, ?, ?, ?
                ))
            ORDER BY TIMESTAMP DESC
            LIMIT 1000
            """,
            [
                self._snowpark_session.get_current_database()[1:-1],
                self._snowpark_session.get_current_schema()[1:-1],
                app_name.upper(),
                app_type,
            ],
            num_expected_spans,
        )

    def _wait_for_num_results(
        self,
        q: str,
        params: List[str],
        expected_num_results: int,
        num_retries: int = 15,
        retry_cooldown_in_seconds: int = 10,
    ) -> List[Row]:
        for _ in range(num_retries):
            results = self.run_query(q, params)

            new_results = []  # TODO(this_pr): remove this
            for curr in results:
                if curr[1] not in [new_curr[1] for new_curr in new_results]:
                    new_results.append(curr)
            results = new_results
            if len(results) == expected_num_results:
                return results
            self.logger.info(
                f"Got {len(results)} results, expecting {expected_num_results}"
            )
            time.sleep(retry_cooldown_in_seconds)
        raise ValueError(
            f"Did not get the expected number of results! Expected {expected_num_results} results, but last found: {len(results)}! The results:\n{results}"
        )
