import json

import pandas as pd
import sqlalchemy as sa
from trulens.apps.app import TruApp
from trulens.core.session import TruSession

import tests.unit.test_otel_tru_custom
from tests.util.otel_test_case import OtelTestCase


class TestOtelAppsTable(OtelTestCase):
    @staticmethod
    def _get_apps() -> pd.DataFrame:
        db = TruSession().connector.db
        with db.session.begin() as db_session:
            q = sa.select(db.orm.AppDefinition)
            return pd.read_sql(q, db_session.bind)

    def test_smoke(self):
        app = tests.unit.test_otel_tru_custom.TestApp()
        TruApp(
            app,
            main_method=app.respond_to_query,
            app_name="MyCustomApp",
            app_version="v1",
        )
        rows = self._get_apps()
        self.assertEqual(1, len(rows))
        self.assertEqual("MyCustomApp", rows["app_name"].iloc[0])
        self.assertEqual("v1", rows["app_version"].iloc[0])
        self.assertGoldenJSONEqual(
            json.loads(rows["app_json"].iloc[0]),
            "tests/unit/data/test_otel_apps_table/test_smoke.json",
            skips=["app_id", "app.__tru_non_serialized_object.id"],
        )
