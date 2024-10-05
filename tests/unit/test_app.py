"""
Tests for TruBasicApp.
"""

from unittest import TestCase
from unittest import main

from trulens.core import session as mod_session
from trulens.core.schema import app as app_schema


class TestApp(TestCase):
    def setUp(self):
        self.session = mod_session.TruSession()
        self.session.reset_database()

        self.app_1 = app_schema.AppDefinition(
            app_name="test_application",
            app_version="v1",
            root_class={"name": "App", "module": {"module_name": "app"}},
            app={},
        )
        self.app_2 = app_schema.AppDefinition(
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
        app_schema.AppDefinition(
            app_name="test_application",
            app_version="v1",
            app_id=self.app_2.app_id,
            root_class={"name": "App", "module": {"module_name": "app"}},
            app={},
        )

        with self.assertWarns(DeprecationWarning):
            # change to below after dep period:
            # with self.assertRaises(ValueError):
            app_schema.AppDefinition(
                app_name="test_application",
                app_version="v1",
                app_id="invalid_app_id",
                root_class={"name": "App", "module": {"module_name": "app"}},
                app={},
            )


if __name__ == "__main__":
    main()
