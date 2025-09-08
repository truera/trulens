from tests.util.otel_test_case import OtelTestCase
from tests.util.snowflake_test_case import SnowflakeTestCase


class OtelSnowflakeTestCase(OtelTestCase, SnowflakeTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        return super().tearDownClass()

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
