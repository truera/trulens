"""
Tests for TruBasicApp.
"""

from unittest import TestCase
from unittest import main

from trulens.core import TruSession
from trulens.core.schema.app import AppDefinition


class TestApp(TestCase):
    def test_app_id(self):
        # Most naive test to make sure the basic app runs at all.
        app_1 = AppDefinition(app_name="test_application", app_version="v1")
        app_2 = AppDefinition(app_name="test_application", app_version="v1")

        self.assertEqual(app_1.app_id, app_2.app_id)

        session = TruSession()
        self.assertEquals(len(session.get_apps()), 1)

        session.add_app(app_1)
        self.assertEquals(len(session.get_apps()), 1)

        AppDefinition(
            app_name="test_application", app_version="v1", app_id=app_2.app_id
        )
        self.assertEquals(len(session.get_apps()), 1)
        with self.assertRaises(ValueError):
            AppDefinition(
                app_name="test_application",
                app_version="v1",
                app_id="invalid_app_id",
            )


if __name__ == "__main__":
    main()
