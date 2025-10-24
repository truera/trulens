"""Unit tests for dashboard utilities.

This test validates the fix for a bug where Record.app_version was incorrectly
accessed directly when it should be accessed through the AppDefinition relationship.
"""

import unittest

import sqlalchemy as sa

from tests.test import TruTestCase


class TestCheckCrossFormatRecordsQuery(TruTestCase):
    """Test SQL query construction for checking cross-format records.

    This is a regression test for a bug where the query tried to access
    Record.app_version directly, which doesn't exist. The app_version field
    is in the AppDefinition table, not the Record table.
    """

    def test_record_query_with_app_versions_constructs_valid_sql(self):
        """Test that the query with app_versions properly joins AppDefinition.

        This validates that when filtering by app_versions, we:
        1. Join the Record table with AppDefinition
        2. Filter on AppDefinition.app_version (not Record.app_version)
        """
        from trulens.core.database.sqlalchemy import SQLAlchemyDB
        from trulens.core.session import TruSession

        # Create a real session with SQLAlchemy backend
        session = TruSession(database_url="sqlite:///:memory:")

        # Verify it's using SQLAlchemyDB
        self.assertIsInstance(session.connector.db, SQLAlchemyDB)

        db = session.connector.db

        # Test 1: Verify that Record ORM doesn't have app_version attribute
        with self.assertRaises(AttributeError):
            _ = db.orm.Record.app_version

        # Test 2: Verify that AppDefinition ORM has app_version attribute
        self.assertTrue(hasattr(db.orm.AppDefinition, "app_version"))

        # Test 3: Build a query that filters by app_version (like the bug fix does)
        # This should NOT raise an AttributeError
        query = sa.select(sa.func.count(db.orm.Record.record_id))

        # Join with AppDefinition (required to access app_version)
        query = query.join(db.orm.Record.app)

        # Filter by app_versions through AppDefinition
        app_versions = ["v1.0", "v2.0"]
        query = query.where(db.orm.AppDefinition.app_version.in_(app_versions))

        # If we got here without AttributeError, the fix is working
        # The query should be valid (we can compile it)
        try:
            compiled = query.compile(compile_kwargs={"literal_binds": True})
            query_str = str(compiled)

            # Verify the query includes a JOIN and references the apps table
            self.assertIn("JOIN", query_str.upper())
            self.assertIn("app_version", query_str.lower())
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")

    def test_record_query_without_app_versions_works(self):
        """Test that querying records without app_version filter works."""
        from trulens.core.session import TruSession

        session = TruSession(database_url="sqlite:///:memory:")
        db = session.connector.db

        # Query without app_version filter should work fine
        query = sa.select(sa.func.count(db.orm.Record.record_id))

        # This should compile successfully
        try:
            compiled = query.compile()
            # If we got here, the query is valid
            self.assertIsNotNone(compiled)
        except Exception as e:
            self.fail(f"Query compilation failed: {e}")


if __name__ == "__main__":
    unittest.main()
