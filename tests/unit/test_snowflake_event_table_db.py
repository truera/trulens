"""Tests for :class:`SnowflakeEventTableDB` that need no Snowflake account."""

from collections import namedtuple
import unittest

import pandas as pd
import pytest

try:
    from trulens.connectors.snowflake.dao.sql_utils import execute_query
    from trulens.connectors.snowflake.snowflake_event_table_db import (
        SnowflakeEventTableDB,
    )

except Exception:
    SnowflakeEventTableDB = None
    execute_query = None

AgentRow = namedtuple("AgentRow", ["name", "kind"])
VersionRow = namedtuple("VersionRow", ["name", "version"])


class FakeSnowparkResult:
    """The result of a ``SHOW`` statement, as snowpark hands it to the DAO."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def collect(self):
        return self.rows

    def to_pandas(self):
        # ``SHOW AGENTS IN ACCOUNT`` is read through snowpark's ``to_pandas``
        # directly, so it keeps the quoted identifiers snowflake reports.
        return pd.DataFrame({'"name"': [row.name for row in self.rows]})


class FakeSession:
    """Answer the ``SHOW`` statements made by ``get_apps``.

    Stands in for a snowpark session on an account holding the given agents.
    """

    def __init__(self, agents=None, versions=None, cortex_agents=None):
        self.agents = agents or []
        self.versions = versions or {}
        self.cortex_agents = cortex_agents or []

    def sql(self, query, params=None):
        if query.startswith("SHOW EXTERNAL AGENTS"):
            return FakeSnowparkResult(self.agents)
        if query.startswith("SHOW VERSIONS IN EXTERNAL AGENT"):
            name = query.split('"')[1]
            return FakeSnowparkResult(self.versions.get(name, []))
        if query.startswith("SHOW AGENTS IN ACCOUNT"):
            return FakeSnowparkResult(self.cortex_agents)
        raise AssertionError(f"Unexpected query {query}")


@pytest.mark.snowflake
class TestSnowflakeEventTableDBGetApps(unittest.TestCase):
    def setUp(self):
        if SnowflakeEventTableDB is None:
            self.skipTest(
                "SnowflakeEventTableDB is not available because snowflake "
                "tests are disabled."
            )

    def test_zero_row_show_statement_yields_no_columns(self):
        """Record what an empty account looks like to its callers.

        ``execute_query`` builds its frame from the rows it got back, so a
        statement with no rows has no ``name`` column at all rather than an
        empty one.
        """
        df = execute_query(FakeSession(), "SHOW EXTERNAL AGENTS;")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        self.assertNotIn("name", df.columns)

    def test_get_apps_lists_agents_that_exist(self):
        session = FakeSession(
            agents=[AgentRow("AGENT_1", "EXTERNAL")],
            versions={
                "AGENT_1": [
                    VersionRow("v1", "LAST"),
                    VersionRow("v2", "DEFAULT"),
                ]
            },
        )
        db = SnowflakeEventTableDB(snowpark_session=session)

        apps = list(db.get_apps())

        self.assertEqual(
            sorted([app["app_name"] for app in apps]),
            ["AGENT_1", "AGENT_1"],
        )
        self.assertEqual(
            sorted([str(app["app_version"]) for app in apps]),
            ["v1", "v2"],
        )

    def test_get_apps_without_external_agents_yields_nothing(self):
        """See [DB.get_apps][trulens.core.database.base.DB.get_apps]."""
        db = SnowflakeEventTableDB(snowpark_session=FakeSession())

        self.assertEqual(list(db.get_apps()), [])

    def test_get_apps_skips_an_agent_that_reports_no_versions(self):
        """A versionless agent is skipped rather than ending the listing.

        ``create_agent_if_not_exist`` treats an empty version listing as a
        state it has to handle, so the app listing has to tolerate the same
        shape from ``SHOW VERSIONS``.
        """
        session = FakeSession(
            agents=[
                AgentRow("AGENT_1", "EXTERNAL"),
                AgentRow("AGENT_2", "EXTERNAL"),
            ],
            versions={"AGENT_2": [VersionRow("v1", "DEFAULT")]},
        )
        db = SnowflakeEventTableDB(snowpark_session=session)

        apps = list(db.get_apps())

        self.assertEqual([str(app["app_name"]) for app in apps], ["AGENT_2"])
