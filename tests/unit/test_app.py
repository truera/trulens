"""
Tests for TruBasicApp.
"""

from unittest import TestCase
from unittest import main

from trulens.core import TruSession
from trulens.core.schema.app import AppDefinition


class TestApp(TestCase):
    def setUp(self):
        self.session = TruSession()
        self.session.reset_database()

        self.app_1 = AppDefinition(
            app_name="test_application",
            app_version="v1",
            root_class={"name": "App", "module": {"module_name": "app"}},
            app={},
        )
        self.app_2 = AppDefinition(
            app_name="test_application",
            app_version="v1",
            root_class={"name": "App", "module": {"module_name": "app"}},
            app={},
        )

    def test_deterministic_app_id(self):
        # Most naive test to make sure the basic app runs at all.
        self.assertEqual(self.app_1.app_id, self.app_2.app_id)

    def test_db_primary_key(self):
        self.session.add_app(self.app_1)
        self.assertEqual(len(self.session.get_apps()), 1)

        self.session.add_app(self.app_2)
        self.assertEqual(len(self.session.get_apps()), 1)

    def test_app_id_override(self):
        AppDefinition(
            app_name="test_application",
            app_version="v1",
            app_id=self.app_2.app_id,
            root_class={"name": "App", "module": {"module_name": "app"}},
            app={},
        )

        with self.assertRaises(ValueError):
            AppDefinition(
                app_name="test_application",
                app_version="v1",
                app_id="invalid_app_id",
                root_class={"name": "App", "module": {"module_name": "app"}},
                app={},
            )


if __name__ == "__main__":
    main()
