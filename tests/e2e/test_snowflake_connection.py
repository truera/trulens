"""
Tests for a Snowflake connection.
"""

from unittest import main

from tests.unit.utils import optional_test
from tests.util.snowflake_test_case import SnowflakeTestCase


class TestSnowflakeConnection(SnowflakeTestCase):
    @optional_test
    def test_basic_snowflake_connection(self):
        """
        Check that we can connect to a Snowflake backend and have created the required schema.
        """
        self.get_tru("test_basic_snowflake_connection")


if __name__ == "__main__":
    main()
